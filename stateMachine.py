"""競技の段階・キー操作・停止条件を管理する。GPIOはここでは操作しない。"""
import math
import numpy as np

# 元からある動作名と関数の引数は維持する。
IDLE, FORWARD, BACKWARD, RIGHT, LEFT = 0, 1, 2, 3, 4

# 以下は実測値ではなく、低速試走を始めるための調整値。
RACE_SECONDS = 180.0
STOP_MARGIN = 0.2
DRIVE_DUTY = 55.0
SLOW_DUTY = 38.0
TURN_DUTY = 35.0
MANUAL_DUTY = 60.0
STEER_GAIN = 35.0
MAX_DUTY = 75.0
LEFT_TRIM = 0.0
RIGHT_TRIM = 0.0
IMAGE_MAX_AGE = 0.5
ENEMY_HOLD_SECONDS = 1.0
SEARCH_SECONDS = 5.0
STALL_SECONDS = 2.5
MAX_RECOVERIES = 3
PASS_CLEAR_SECONDS = 0.4
GOAL_NEAR_AREA = 0.025
GOAL_CREEP_AREA = 0.065
# ゴール位置では赤は見えない。最後に見失った位置からの前進時間を校正する。
# Noneの間は見失った時点で停止する。距離を推測して押し込まない。
GOAL_FINAL_ADVANCE_SECONDS = None
GOAL_FINAL_DUTY = 38.0
GOAL_FINAL_MAX_SECONDS = 1.0
GOAL_CREEP_LIMIT = 1.0


def _target_x(info):
    if info is None:
        return None
    if isinstance(info, dict):
        x = info.get("x", -1)
    else:
        # 旧形式は320画素幅の (x, y, size)。新しい制御は辞書を使う。
        if len(info) < 3 or info[0] < 0 or info[2] <= 0:
            return None
        x = info[0] / 320.0
    return float(x) if math.isfinite(x) and 0 <= x <= 1 else None


def stateMachine(sState, vFlagInfo, vEnemyInfo):
    """目標の左右から基本動作を選ぶ。無効値の確認を位置判定より先にする。"""
    x = _target_x(vFlagInfo)
    enemy_x = _target_x(vEnemyInfo)
    if x is None:
        return IDLE
    if enemy_x is not None and abs(enemy_x - 0.5) < 0.25:
        return IDLE
    # 旋回開始と終了の幅を変えると、境界付近の左右の震えを減らせる。
    tolerance = 0.02 if sState in (LEFT, RIGHT) else 0.05
    if x > 0.5 + tolerance:
        return RIGHT
    if x < 0.5 - tolerance:
        return LEFT
    return FORWARD


class RaceController:
    """mode＝操作方法、phase＝開始待ち等、stage＝コースの進行段階。"""

    def __init__(self):
        self.mode = 1
        self.phase = "MANUAL"
        self.stage = "START_EXIT"
        self.route = None
        self.started_at = None
        self.deadline = None
        self.pending_start = None
        self.auto_stopped = False
        self.quit = False
        self.command = (0.0, 0.0)  # (左PWM, 右PWM)。負なら後退。
        self.manual_command = self.command
        self.motion = IDLE
        self.reason = "手動: w前進 s後退 a左回転 d右回転 e停止 / 2自律"
        self.stage_entered = None
        self.last_progress = None
        self.last_motion_sample = None
        self.previous_gray = None
        self.previous_mask = None
        self.forward_evidence = 0
        self.last_clear_motion = -math.inf
        self.last_turn = -math.inf
        self.last_enemy = -math.inf
        self.enemy_until = -math.inf
        self.track = None
        self.search_started = None
        self.recovery = None
        self.recoveries = 0
        self.near_goal_confirmations = 0
        self.creep_started = None
        self.last_goal_seen = -math.inf
        self.last_goal_x = 0.5
        self.last_goal_area = 0.0
        self.final_driven_seconds = 0.0
        self.final_deadline = None
        self.last_update_at = None

    def stop(self, reason, terminal=True):
        self.command = (0.0, 0.0)
        self.manual_command = self.command
        self.reason = reason
        self.pending_start = None
        if terminal:
            self.auto_stopped = True
            self.phase = "STOPPED"

    def check_deadline(self, now):
        if self.deadline is not None and now >= self.deadline:
            # 手動は時間制限の対象外。ただし自律の時計を延長はしない。
            if self.mode == 1:
                self.auto_stopped = True
                return False
            self.stop("180秒の停止期限（再試行はプログラムを再起動）")
            return True
        return False

    def handle_key(self, key, now):
        """sは手動なら後退、自律の開始待ちなら開始要求になる。"""
        if isinstance(key, int):
            key = chr(key) if 0 <= key < 128 else ""
        if key == "q":
            self.quit = True
            self.stop("終了")
            return
        if key == "e":
            self.stop("eキーで停止", terminal=self.mode == 2 and
                      self.started_at is not None)
            return
        if key in ("1", "2"):
            new_mode = int(key)
            if new_mode != self.mode:
                # 視覚を失った最後の区間は、途中からの距離を復元できない。
                interrupted_final = self.mode == 2 and self.stage == "GOAL_FINAL"
                self.stop("モード切替で停止", terminal=interrupted_final)
                self.mode = new_mode
                if self.mode == 1:
                    self.phase = "MANUAL"
                elif self.auto_stopped:
                    self.phase = "STOPPED"
                else:
                    self.phase = "READY" if self.route else "SELECT"
                    self.reason = "r右側 / l左側を選択後、sで開始"
            self.check_deadline(now)
            return  # 同じ2を押しても時刻や段階を初期化しない。
        if self.check_deadline(now):
            return
        if self.mode == 1:
            self.manual_command = {
                "w": (MANUAL_DUTY, MANUAL_DUTY),
                "s": (-MANUAL_DUTY, -MANUAL_DUTY),
                "a": (-TURN_DUTY, TURN_DUTY),
                "d": (TURN_DUTY, -TURN_DUTY),
            }.get(key, self.manual_command)
        elif not self.auto_stopped:
            if key in ("r", "l") and self.started_at is None:
                self.route = "right" if key == "r" else "left"
                self.phase = "READY"
                self.reason = self.route + "を選択。sで開始"
            elif key == "s" and self.phase == "READY":
                self.pending_start = now  # 有効画像を確認したupdateで受理する。
            elif key == "s" and not self.route:
                self.reason = "先にrまたはlで通行側を選択してください"

    def sensor_failure(self, now):
        self.stop("画像が無効。停止して再取得",
                  terminal=self.mode == 2 and self.stage == "GOAL_FINAL")
        self.previous_gray = None
        self.search_started = None
        self.recovery = None
        self.last_clear_motion = -math.inf
        self.near_goal_confirmations = 0
        # 取得異常の間を走行時間・通過実績として数えない。
        if self.track:
            self.track["edge_at"] = None
            self.track["evidence_at_edge"] = self.forward_evidence
        self.last_progress = now
        self.check_deadline(now)

    def _motion_changed(self, scene, now):
        if (self.last_motion_sample is not None and
                now - self.last_motion_sample < 0.15):
            return False
        gray = scene.get("motion_gray")
        mask = scene.get("motion_mask")
        if gray is None or mask is None:
            return False
        changed = False
        if self.previous_gray is not None:
            valid = mask & self.previous_mask
            if np.count_nonzero(valid) > 200:
                difference = np.abs(gray - self.previous_gray)
                changed = bool(np.mean(difference[valid]) > 0.018)
        self.previous_gray = gray.copy()
        self.previous_mask = mask.copy()
        self.last_motion_sample = now
        return changed

    def _drive_toward(self, x, duty):
        self.motion = stateMachine(self.motion, {"x": x}, None)
        error = (x - 0.5) * 2
        turn = STEER_GAIN * error
        base = duty * (1 - 0.35 * abs(error))
        # 通常の追従では後退させず、前進の左右差で曲がる。
        left = min(MAX_DUTY, max(0, base + turn + LEFT_TRIM))
        right = min(MAX_DUTY, max(0, base - turn + RIGHT_TRIM))
        return left, right

    def _rotate(self, side):
        if side == "left":
            return -TURN_DUTY, TURN_DUTY
        return TURN_DUTY, -TURN_DUTY

    def _change_stage(self, stage, now):
        self.stage = stage
        self.stage_entered = now
        self.track = None
        self.search_started = None
        self.recoveries = 0
        self.last_progress = now
        self.reason = "段階: " + stage

    def _enemy_blocks(self, scene, now):
        for enemy in scene["enemies"]:
            self.last_enemy = now
            # 目印より車体の方が広いので、黄色領域の両側に余裕を足す。
            overlaps = enemy["left"] < 0.72 and enemy["right"] > 0.28
            near = enemy["height"] > 0.08 or enemy["area"] > 0.006
            if overlaps and near:
                self.enemy_until = now + ENEMY_HOLD_SECONDS
        return now < self.enemy_until

    @staticmethod
    def _nearest(objects):
        return max(objects, key=lambda obj: obj["bottom"] + obj["height"],
                   default=None)

    def _pass_object(self, objects, next_stage, scene, now, forced_side=None):
        """接近・側方への移動・前進の証拠をそろえて通過を推定する。"""
        obj = self._nearest(objects)
        if self.track is None and obj is not None:
            side = forced_side or ("left" if obj["x"] >= 0.5 else "right")
            self.track = {"object": obj, "side": side, "near": False,
                          "edge_at": None, "evidence_at_edge": 0}
        track = self.track
        if track is None:
            return None
        if obj is not None:
            previous = track["object"]
            # 遠方にある別の緑円柱へ追跡対象を乗り換えない。
            if (abs(obj["x"] - previous["x"]) > 0.3 or
                    obj["height"] < previous["height"] * 0.5):
                obj = None
        if obj is not None:
            track["object"] = obj
            track["near"] |= obj["bottom"] > 0.65 and obj["height"] > 0.16
            at_edge = (obj["x"] > 0.78 if track["side"] == "left"
                       else obj["x"] < 0.22)
            if track["near"] and at_edge and track["edge_at"] is None:
                track["edge_at"] = now
                track["evidence_at_edge"] = self.forward_evidence
        previous = track["object"]
        passed = (obj is None and track["edge_at"] is not None and
                  now - track["edge_at"] >= PASS_CLEAR_SECONDS and
                  self.forward_evidence - track["evidence_at_edge"] >= 2)
        if passed:
            self._change_stage(next_stage, now)
            return (0.0, 0.0)  # 段階変更時は一度止め、次の画像で進路を確認。
        if obj is None and track["near"] and track["edge_at"] is None:
            self.reason = "接近対象を前方で喪失。通過と決めつけず停止"
            return (0.0, 0.0)
        # 見えている物体の外縁より外に目標を置き、車体幅分の余裕を取る。
        x = previous["left"] - 0.16 if track["side"] == "left" else previous["right"] + 0.16
        x = min(0.92, max(0.08, x))
        if obj is None:
            # 消失後は以前の強い舵を続けず、短い直進で通過の証拠を確認。
            x = 0.5
        self.reason = "指定側でコーン通過" if forced_side else "緑円柱を回避"
        return self._drive_toward(x, SLOW_DUTY)

    def _search(self, now):
        if self.search_started is None:
            self.search_started = now
        elapsed = now - self.search_started
        if elapsed >= SEARCH_SECONDS:
            self.stop("探索上限。目標を再取得できないため停止")
            return self.command
        # 最初は最後に見た側、その後は左右を交互に小さく探す。
        side = "right" if self.last_goal_x > 0.5 else "left"
        if int(elapsed / 0.8) % 2:
            side = "left" if side == "right" else "right"
        self.reason = "目標探索（時間上限あり）"
        return self._rotate(side)

    def _begin_recovery(self, scene, now):
        self.recoveries += 1
        if self.recoveries > MAX_RECOVERIES:
            self.stop("自己復帰回数の上限")
            return
        # 後方センサはない。最近まっすぐ通った区間の証拠があるときだけ戻る。
        rear_evidence = (self.stage != "START_EXIT" and
                         now - self.last_clear_motion < 3.0 and
                         now - self.last_turn > 2.0 and
                         now - self.last_enemy > 3.0)
        nearby = any(obj["bottom"] > 0.8 for obj in
                     scene["obstacles"] + scene["cones"])
        if nearby and not rear_evidence:
            self.stop("周囲が狭く後方も不明。無条件の後退・旋回はしない")
            return
        self.recovery = {"started": now, "reverse": rear_evidence}
        self.command = (0.0, 0.0)
        self.reason = "停滞を検出。一旦停止して自己復帰"

    def _recover(self, now):
        elapsed = now - self.recovery["started"]
        if elapsed < 0.15:
            return 0.0, 0.0
        if self.recovery["reverse"] and elapsed < 0.33:
            return -SLOW_DUTY, -SLOW_DUTY
        if elapsed < 0.75:
            side = self.route
            if self.recoveries % 2 == 0:
                side = "left" if side == "right" else "right"
            return self._rotate(side)
        self.recovery = None
        self.track = None  # 復帰で姿勢が変わったので、通過の証拠は集め直す。
        self.last_progress = now
        return 0.0, 0.0

    def _final_goal(self):
        duration = GOAL_FINAL_ADVANCE_SECONDS
        if duration is None or not (0 < duration <= GOAL_FINAL_MAX_SECONDS):
            self.stop("赤を見失った。最終前進時間が未校正のため停止")
            return self.command
        if self.final_driven_seconds >= duration:
            self.stop("校正した最終前進を完了。ゴール到達は推定（eで停止）")
            self.phase = "GOAL_HOLD"
            return self.command
        self.reason = "赤は視野外。校正した短い最終前進"
        return self._drive_toward(0.5, GOAL_FINAL_DUTY)

    def _goal(self, flag, now):
        if self.stage == "GOAL_FINAL":
            if flag is not None:
                # 遮蔽で消えただけの可能性がある。距離校正の起点を取り直す。
                self._change_stage("GOAL_ALIGN", now)
                self.near_goal_confirmations = 0
                self.final_deadline = None
                return 0.0, 0.0
            return self._final_goal()
        if flag is None:
            near_and_aligned = (self.near_goal_confirmations >= 3 and
                                self.last_goal_area >= GOAL_NEAR_AREA and
                                abs(self.last_goal_x - 0.5) < 0.07 and
                                now - self.last_goal_seen <= IMAGE_MAX_AGE)
            if near_and_aligned:
                # 消失はゴール成立ではなく、「最後の見えない区間」の開始。
                self._change_stage("GOAL_FINAL", now)
                self.final_driven_seconds = 0.0
                if GOAL_FINAL_ADVANCE_SECONDS is not None:
                    self.final_deadline = now + GOAL_FINAL_ADVANCE_SECONDS
                return self._final_goal()
            if self.stage == "GOAL_CREEP" or self.last_goal_area >= GOAL_NEAR_AREA:
                self.stop("ゴール付近の消失を確認できないため停止")
                return self.command
            return self._search(now)
        error = abs(flag["x"] - 0.5)
        if flag["area"] >= GOAL_NEAR_AREA and error < 0.07:
            self.near_goal_confirmations += 1
        else:
            self.near_goal_confirmations = 0
        if self.stage == "GOAL_ALIGN":
            if error < 0.05:
                self._change_stage("GOAL_APPROACH", now)
            return self._drive_toward(flag["x"], SLOW_DUTY)
        if self.stage == "GOAL_APPROACH":
            if flag["area"] >= GOAL_CREEP_AREA and error < 0.07:
                self._change_stage("GOAL_CREEP", now)
                self.creep_started = now
            duty = SLOW_DUTY if flag["area"] >= GOAL_NEAR_AREA else DRIVE_DUTY
            return self._drive_toward(flag["x"], duty)
        elapsed = now - self.creep_started
        if elapsed >= GOAL_CREEP_LIMIT:
            self.stop("赤が見える区間の微速進入上限。到達未確認")
            return self.command
        # 短く進み、止まって観測する。sleepで入力処理を止めない。
        self.reason = "ゴール微速進入（到達は画像による推定）"
        if elapsed % 0.35 >= 0.12:
            return 0.0, 0.0
        return self._drive_toward(flag["x"], SLOW_DUTY)

    def update(self, scene, now):
        """有効画像1枚につき1回呼び、(左PWM, 右PWM)を返す。"""
        dt = 0 if self.last_update_at is None else now - self.last_update_at
        self.last_update_at = now
        if self.check_deadline(now) or self.quit:
            return self.command
        if scene is None or not scene.get("valid", False) or now - scene.get("received_at", now) > IMAGE_MAX_AGE:
            self.sensor_failure(now)
            return self.command
        moving = self._motion_changed(scene, now)
        if self.mode == 1:
            self.command = self.manual_command
            return self.command
        if self.auto_stopped:
            self.command = (0.0, 0.0)
            return self.command
        if self.stage == "GOAL_FINAL" and self.phase == "RUNNING":
            if dt > IMAGE_MAX_AGE:
                self.stop("最終進入中に観測更新が遅延。移動量が不明のため停止")
                return self.command
            if min(self.command) > 0:
                self.final_driven_seconds += max(0.0, dt)
        if self.pending_start is not None:
            if self.started_at is None:
                self.started_at = self.pending_start
                self.deadline = self.started_at + RACE_SECONDS - STOP_MARGIN
                self.stage_entered = now
            self.pending_start = None
            self.phase = "RUNNING"
            self.last_progress = now
            self.last_motion_sample = now
            self.track = None
            self.recovery = None
            self.search_started = None
            if self.check_deadline(now):
                return self.command
        if self.phase != "RUNNING":
            self.command = (0.0, 0.0)
            return self.command
        if moving and min(self.command) > 0:
            self.forward_evidence += 1
            self.last_progress = now
            if abs(self.command[0] - self.command[1]) < 10:
                if not scene["enemies"] and not scene["obstacles"] and not scene["cones"]:
                    self.last_clear_motion = now
        if abs(self.command[0] - self.command[1]) > 15:
            self.last_turn = now
        flag = scene["flag"]
        if flag:
            self.last_goal_x = flag["x"]
            self.last_goal_seen = now
            self.last_goal_area = flag["area"]
        if self._enemy_blocks(scene, now):
            if self.stage == "GOAL_FINAL":
                self.stop("最終進入中に相手を検出。到達未確認で停止")
                return self.command
            self.command = (0.0, 0.0)
            self.reason = "相手の接近／通過待ち"
            self.last_progress = now  # 意図した待機をスタックと判定しない。
            self.near_goal_confirmations = 0
            # 相手による遮蔽中に通過時間だけを消費しない。
            if self.track:
                self.track["edge_at"] = None
            return self.command
        if self.recovery:
            self.command = self._recover(now)
            return self.command
        if (self.stage not in ("GOAL_CREEP", "GOAL_FINAL") and
                self.search_started is None and min(self.command) > 0 and
                now - self.last_progress >= STALL_SECONDS):
            self._begin_recovery(scene, now)
            return self.command
        self.search_started = None if flag else self.search_started
        if self.stage == "START_EXIT":
            if self.forward_evidence >= 2 and now - self.stage_entered >= 0.4:
                self._change_stage("NEAR_OBSTACLE", now)
        if self.stage in ("NEAR_OBSTACLE", "FAR_OBSTACLE"):
            next_stage = "CENTER_APPROACH" if self.stage == "NEAR_OBSTACLE" else "GOAL_ALIGN"
            command = self._pass_object(scene["obstacles"], next_stage, scene, now)
        elif self.stage in ("CENTER_APPROACH", "CENTER_PASS"):
            if scene["cones"]:
                self.stage = "CENTER_PASS"
            command = self._pass_object(scene["cones"], "FAR_OBSTACLE", scene,
                                        now, forced_side=self.route)
        elif self.stage.startswith("GOAL_"):
            command = self._goal(flag, now)
        else:
            command = None
        if command is None:
            if flag:
                command = self._drive_toward(flag["x"], DRIVE_DUTY)
            elif self.stage == "START_EXIT" and scene["obstacles"]:
                # 遠方の赤が小さくても、見えている手前円柱を避けて離脱できる。
                obj = self._nearest(scene["obstacles"])
                x = obj["left"] - 0.16 if obj["x"] >= 0.5 else obj["right"] + 0.16
                command = self._drive_toward(min(0.92, max(0.08, x)), SLOW_DUTY)
            else:
                command = self._search(now)
        # 通過対象以外の目前の物体も、進路上なら停止する。
        near_objects = scene["obstacles"] + scene["cones"]
        center_blocked = any(obj["bottom"] > 0.92 and obj["height"] > 0.2 and
                             obj["left"] < 0.58 and obj["right"] > 0.42
                             for obj in near_objects)
        if center_blocked and min(command) > 0:
            command = (0.0, 0.0)
            self.reason = "目前の障害物。接触を避けて停止"
            if self.stage == "GOAL_FINAL":
                self.stop("最終進入の前方が塞がったため停止")
        self.command = command
        return self.command

"""競技の段階・キー操作・停止条件を管理する。GPIOはここでは操作しない。"""
import math
import numpy as np

# 基本動作の識別番号。順に停止・前進・後退・右旋回・左旋回を表す。
IDLE, FORWARD, BACKWARD, RIGHT, LEFT = 0, 1, 2, 3, 4

# 自律走行の時間設定（秒）。手動モードは制限時間の対象外。
# 開始から RACE_SECONDS - STOP_MARGIN 秒で停止指令を出す。
RACE_SECONDS = 180.0  # 競技の制限時間。
STOP_MARGIN = 0.2  # 監視周期や車体が止まるまでの遅れを見込んで早めに止める時間。

# モータへのPWM指令。0～100%のデューティ比で、実際の速度は機体によって異なる。
# 走行用の値は低速試走の初期値。実機で動き出す最低PWMも確認して調整する。
DRIVE_DUTY = 100.0  # 自律走行で通常前進するときの基本PWM（%）。
SLOW_DUTY = 80.0  # 障害物回避・ゴール接近・復帰時の短い後退に使うPWM（%）。
TURN_DUTY = 80.0  # 左右のモータを逆向きに回す旋回のPWM（%）。手動と自律で共通。
MANUAL_DUTY = 100.0  # 手動で前進・後退するときのPWM（%）。
STEER_GAIN = 10.0  # 目標の横ずれが最大のとき、片側に足し引きするPWMのポイント数。大きいほど強く曲がる。
MAX_DUTY = 100.0  # 左右それぞれの出力上限（%）。手動時の出力にも適用される。
LEFT_TRIM = 0.0  # 自律で目標を追うときの左モータ補正。正なら左PWMを増やす（ポイント）。
RIGHT_TRIM = 0.0  # 自律で目標を追うときの右モータ補正。正なら右PWMを増やす（ポイント）。

# 画像の有効期限、待機・復帰・通過の設定。
# 画像の座標は左端・上端を0、右端・下端を1とする。物体の高さや面積も画像全体に対する割合で扱う。
IMAGE_MAX_AGE = 0.5  # 画像取得から判断までの許容時間（秒）。最終進入の更新間隔・赤の消失判定にも使う。
ENEMY_HOLD_SECONDS = 1.0  # 進路上の相手を最後に検出してから停止を続ける時間（秒）。
SEARCH_SECONDS = 5.0  # 目標を見失ったときに左右を探索する時間の上限（秒）。
STALL_SECONDS = 2.5  # 前進指令中に移動の証拠が得られず、復帰を試すまでの時間（秒）。
MAX_RECOVERIES = 3  # 同じコース段階での復帰試行回数の上限。段階が変わるとリセット。
PASS_CLEAR_SECONDS = 0.4  # 物体が側方に抜け始めてから、通過と判断するまでの最低時間（秒）。
OBSTACLE_CLEARANCE_X = 0.08  # 緑円柱の外縁から取る余裕。初期値は目前の接触判定と同じく中央の左右8%。
OBSTACLE_PASS_BOTTOM = 0.90  # 横に十分離れた円柱の下端がこの高さを超えたら、通過の確認を始める。
CONE_SIDE_MARGIN_X = 0.16  # コーンの外縁から確保する横方向の余裕。画像幅に対する割合。
CONE_CLEAR_SECONDS = 1.0  # コーンが視野から消えた後に直進する最低時間（秒）。SLOW_DUTYで試走して調整する。

# ゴールの判定に使う面積は「赤い輪郭の面積 ÷ 処理画像全体の面積」。
GOAL_NEAR_AREA = 0.025  # 面積比2.5%。減速と、赤を見失う直前に十分近かったかの判定に使う。
GOAL_CREEP_AREA = 0.065  # 面積比6.5%。赤が中央付近にあれば微速進入に切り替える。
# ゴール位置では赤が視野から外れる。最後の前進時間は、下のPWMで実機を走らせて測る。
GOAL_FINAL_ADVANCE_SECONDS = None  # 消失後に前進する時間（秒）。未設定（None）なら停止。
GOAL_FINAL_DUTY = 100.0  # 赤を近接で見失った後の前進の基本PWM（%）。変更したら時間も測り直す。
GOAL_FINAL_MAX_SECONDS = 1.0  # 視野外での前進時間に設定できる上限（秒）。
GOAL_CREEP_LIMIT = 1.0  # 赤がまだ見える微速区間の上限（秒）。停止して観測する時間も含む。


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
    # xは画像左端が0、中央が0.5、右端が1。中央から左右25%幅に相手がいれば停止。
    if enemy_x is not None and abs(enemy_x - 0.5) < 0.25:
        return IDLE
    # 目標が中央から5%幅を超えてずれると旋回し、2%幅以内まで戻ると前進する。
    # 開始と終了の幅を分けると、境界付近で前進と旋回を繰り返しにくい。
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
        self.cone_clear_elapsed = None  # コーンが消えてから直進した時間。Noneは消失をまだ確認していない状態。
        self.cone_evidence_at_loss = 0
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
        self.cone_clear_elapsed = None
        self.last_progress = now
        self.check_deadline(now)

    def _motion_changed(self, scene, now):
        # 移動の比較は最短0.15秒ごと。直前のフレームとの差が小さすぎるのを防ぐ。
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
            # 両画像で比較できる背景が200画素を超える場合だけ調べる。
            # 明るさは0～1。平均の差が0.018（明るさの全範囲の1.8%）を超えたら移動の証拠にする。
            if np.count_nonzero(valid) > 200:
                difference = np.abs(gray - self.previous_gray)
                changed = bool(np.mean(difference[valid]) > 0.018)
        self.previous_gray = gray.copy()
        self.previous_mask = mask.copy()
        self.last_motion_sample = now
        return changed

    def _drive_toward(self, x, duty):
        self.motion = stateMachine(self.motion, {"x": x}, None)
        error = (x - 0.5) * 2  # 中央からのずれを左端-1、右端+1に換算する。
        turn = STEER_GAIN * error
        base = duty * (1 - 0.35 * abs(error))  # ずれが大きいほど減速し、端では基本PWMを35%下げる。
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
            # 黄色い領域が画像幅の28～72%に重なれば、進路にかかるとみなす。
            overlaps = enemy["left"] < 0.72 and enemy["right"] > 0.28
            # 目印が画像の高さの8%、または画像全体の面積の0.6%を超えたら近いとみなす。
            near = enemy["height"] > 0.08 or enemy["area"] > 0.006
            if overlaps and near:
                self.enemy_until = now + ENEMY_HOLD_SECONDS
        return now < self.enemy_until

    @staticmethod
    def _nearest(objects):
        return max(objects, key=lambda obj: obj["bottom"] + obj["height"],
                   default=None)

    @staticmethod
    def _obstacle_target(obj, side):
        # 円柱と余裕を含む範囲から、画面中央（現在の進行方向）を外すための最小限の舵。
        # すでに中央が空いていれば、左右どちらに回避中でも0.5に戻して直進する。
        if side == "left":
            return max(0.08, min(0.5, obj["left"] - OBSTACLE_CLEARANCE_X))
        return min(0.92, max(0.5, obj["right"] + OBSTACLE_CLEARANCE_X))

    def _pass_object(self, objects, next_stage, now):
        """接近・側方への移動・前進の証拠をそろえて通過を推定する。"""
        obj = self._nearest(objects)
        if self.track is None and obj is not None:
            side = "left" if obj["x"] >= 0.5 else "right"
            self.track = {"object": obj, "side": side, "near": False,
                          "edge_at": None, "evidence_at_edge": 0}
        track = self.track
        if track is None:
            return None
        x = 0.5  # 対象がいったん消えた場合は、それまでの方位を保つ。
        if obj is not None:
            previous = track["object"]
            # 中心が画像幅の30%を超えて跳ぶか、高さが前回の半分未満なら別の物体とみなす。
            if (abs(obj["x"] - previous["x"]) > 0.3 or
                    obj["height"] < previous["height"] * 0.5):
                obj = None
        if obj is not None:
            track["object"] = obj
            # 物体の下端が画像の縦65%より下で、高さが画像の16%を超えたら接近済みと記録。
            track["near"] |= obj["bottom"] > 0.65 and obj["height"] > 0.16
            # 左を通ると物体は右端の22%に入り、右を通ると左端の22%に入る。
            at_edge = (obj["x"] > 0.78 if track["side"] == "left"
                       else obj["x"] < 0.22)
            x = self._obstacle_target(obj, track["side"])
            # 横に余裕がある円柱は、画面下端から見切れることもある。
            at_edge = x == 0.5 and (at_edge or obj["bottom"] > OBSTACLE_PASS_BOTTOM)
            if x != 0.5:
                track["edge_at"] = None  # 円柱が進路に戻ったら通過の確認をやり直す。
            if track["near"] and at_edge and track["edge_at"] is None:
                track["edge_at"] = now
                track["evidence_at_edge"] = self.forward_evidence
        # 物体の消失と経過時間に加え、側方に抜け始めた後の前進の証拠が2回以上必要。
        passed = (obj is None and track["edge_at"] is not None and
                  now - track["edge_at"] >= PASS_CLEAR_SECONDS and
                  self.forward_evidence - track["evidence_at_edge"] >= 2)
        if passed:
            self._change_stage(next_stage, now)
            return (0.0, 0.0)  # 段階変更時は一度止め、次の画像で進路を確認。
        if obj is None and track["near"] and track["edge_at"] is None:
            self.reason = "接近対象を前方で喪失。通過と決めつけず停止"
            return (0.0, 0.0)
        self.reason = "緑円柱を回避"
        return self._drive_toward(x, SLOW_DUTY)

    def _pass_cone(self, cones, now, dt):
        """指定側へ進入し、コーンが視野外に出た後も後部が抜けるまで直進する。"""
        cone = self._nearest(cones)
        if cone is not None:
            # コーンが再び見えたら、消失後の直進時間は最初から数える。
            self.cone_clear_elapsed = None
            if self.route == "left":
                boundary = cone["left"] - CONE_SIDE_MARGIN_X
                clear_side = boundary >= 0.5  # コーンの左端が進路より十分右にある。
            else:
                boundary = cone["right"] + CONE_SIDE_MARGIN_X
                clear_side = boundary <= 0.5
            if self.stage == "CENTER_CLEAR":
                if not clear_side:
                    self.reason = "コーンが直進範囲に入ったため停止"
                    return 0.0, 0.0
            else:
                near = cone["bottom"] > 0.65 and cone["height"] > 0.16
                if not (near and clear_side):
                    self.stage = "CENTER_PASS"
                    self.reason = "指定側でコーンに進入"
                    # 進入時の余裕幅は従来と同じ。余裕がある場合はコーン側へ切り込まない。
                    x = 0.5 if clear_side else min(0.92, max(0.08, boundary))
                    return self._drive_toward(x, SLOW_DUTY)
                self._change_stage("CENTER_CLEAR", now)
        elif self.stage == "CENTER_APPROACH":
            return None  # まだコーンを見ていない間は、通常の追従で接近する。
        elif self.stage != "CENTER_CLEAR":
            self.reason = "コーンの側方を確認できず停止。再検出を待つ"
            return 0.0, 0.0
        else:
            if self.cone_clear_elapsed is None or dt > IMAGE_MAX_AGE:
                # 消失を最初に観測した時点を起点にする。それ以前の時間は含めない。
                self.cone_clear_elapsed = 0.0
                self.cone_evidence_at_loss = self.forward_evidence
            elif min(self.command) > 0:
                self.cone_clear_elapsed += dt  # 直前の指令が前進の場合だけ時間を足す。
            if (self.cone_clear_elapsed >= CONE_CLEAR_SECONDS and
                    self.forward_evidence - self.cone_evidence_at_loss >= 2):
                self._change_stage("FAR_OBSTACLE", now)
                return 0.0, 0.0  # 次の画像から奥の障害物に向けて操舵する。

        self.reason = "コーンの横を直進。車体後部が抜けるまで進路を維持"
        return self._drive_toward(0.5, SLOW_DUTY)

    def _search(self, now):
        if self.search_started is None:
            self.search_started = now
        elapsed = now - self.search_started
        if elapsed >= SEARCH_SECONDS:
            self.stop("探索上限。目標を再取得できないため停止")
            return self.command
        # 最初は目標を最後に見た側へ向き、その後は0.8秒ごとに旋回方向を変える。
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
        # 直進の証拠が3秒以内にあり、直近2秒に大きな旋回、直近3秒に相手の検出がなければ短く後退できる。
        # 出発直後は自分側のポールが後ろにあるため、後退の候補から外す。
        rear_evidence = (self.stage != "START_EXIT" and
                         now - self.last_clear_motion < 3.0 and
                         now - self.last_turn > 2.0 and
                         now - self.last_enemy > 3.0)
        # 物体の下端が画像の縦80%より下なら、付近に障害物があるとみなす。
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
        if elapsed < 0.15:  # 復帰開始から0.15秒間は停止。
            return 0.0, 0.0
        if self.recovery["reverse"] and elapsed < 0.33:  # 後退できる場合は開始0.33秒まで後退。
            return -SLOW_DUTY, -SLOW_DUTY
        if elapsed < 0.75:  # 開始0.75秒まで旋回。失敗が続いたら試行ごとに向きを変える。
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
            # 近接と整列を3回以上確認し、最後の赤が中央から7%幅未満で、観測も新しければ最終進入へ。
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
        # 中央から7%幅未満で近接面積も満たすフレームを数える。外れたら数え直す。
        if flag["area"] >= GOAL_NEAR_AREA and error < 0.07:
            self.near_goal_confirmations += 1
        else:
            self.near_goal_confirmations = 0
        if self.stage == "GOAL_ALIGN":
            if error < 0.05:  # 最初の整列は中央から5%幅未満で完了。
                self._change_stage("GOAL_APPROACH", now)
            return self._drive_toward(flag["x"], SLOW_DUTY)
        if self.stage == "GOAL_APPROACH":
            # 微速進入には面積条件に加え、目標が中央から7%幅未満にあることが必要。
            if flag["area"] >= GOAL_CREEP_AREA and error < 0.07:
                self._change_stage("GOAL_CREEP", now)
                self.creep_started = now
            duty = SLOW_DUTY if flag["area"] >= GOAL_NEAR_AREA else DRIVE_DUTY
            return self._drive_toward(flag["x"], duty)
        elapsed = now - self.creep_started
        if elapsed >= GOAL_CREEP_LIMIT:
            self.stop("赤が見える区間の微速進入上限。到達未確認")
            return self.command
        # 0.35秒を1周期として、最初の0.12秒だけ前進し、残りの時間は停止して観測する。
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
            if self.stage == "CENTER_CLEAR":
                # 手動の間に姿勢が変わり得るので、再開時はコーンの側方を確かめ直す。
                self.stage = "CENTER_PASS"
                self.cone_clear_elapsed = None
            if self.check_deadline(now):
                return self.command
        if self.phase != "RUNNING":
            self.command = (0.0, 0.0)
            return self.command
        if moving and min(self.command) > 0:
            self.forward_evidence += 1
            self.last_progress = now
            # 左右のPWM差が10ポイント未満で、相手・円柱・コーンが見えなければ直進を記録する。
            if abs(self.command[0] - self.command[1]) < 10:
                if not scene["enemies"] and not scene["obstacles"] and not scene["cones"]:
                    self.last_clear_motion = now
        if abs(self.command[0] - self.command[1]) > 15:  # 15ポイントを超える差は大きな旋回の指令とみなす。
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
            self.cone_clear_elapsed = None
            return self.command
        if self.recovery:
            self.command = self._recover(now)
            return self.command
        if (self.stage not in ("GOAL_CREEP", "GOAL_FINAL") and
                self.search_started is None and min(self.command) > 0 and
                now - self.last_progress >= STALL_SECONDS):
            if self.stage == "CENTER_CLEAR":
                self.stop("コーンの横で前進を確認できないため停止")
            else:
                self._begin_recovery(scene, now)
            return self.command
        self.search_started = None if flag else self.search_started
        if self.stage == "START_EXIT":
            # 発進から0.4秒以上経ち、前進の証拠が2回以上あれば次の段階へ進む。
            if self.forward_evidence >= 2 and now - self.stage_entered >= 0.4:
                self._change_stage("NEAR_OBSTACLE", now)
        if self.stage in ("NEAR_OBSTACLE", "FAR_OBSTACLE"):
            next_stage = "CENTER_APPROACH" if self.stage == "NEAR_OBSTACLE" else "GOAL_ALIGN"
            command = self._pass_object(scene["obstacles"], next_stage, now)
        elif self.stage in ("CENTER_APPROACH", "CENTER_PASS", "CENTER_CLEAR"):
            command = self._pass_cone(scene["cones"], now, dt)
        elif self.stage.startswith("GOAL_"):
            command = self._goal(flag, now)
        else:
            command = None
        if command is None:
            if self.stage == "START_EXIT" and scene["obstacles"]:
                # 発進直後も眼前の円柱を優先し、進路が空いていればそのまま直進する。
                obj = self._nearest(scene["obstacles"])
                side = "left" if obj["x"] >= 0.5 else "right"
                command = self._drive_toward(self._obstacle_target(obj, side), SLOW_DUTY)
            elif flag:
                command = self._drive_toward(flag["x"], DRIVE_DUTY)
            else:
                command = self._search(now)
        # 下端が画像の縦92%より下、高さが20%超で、横42～58%の進路に重なる物体があれば停止。
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

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
SLOW_DUTY = 100.0  # 障害物回避・ゴール接近・復帰時の短い後退に使うPWM（%）。
TURN_DUTY = 100.0  # 左右のモータを逆向きに回す旋回のPWM（%）。手動と自律で共通。
MANUAL_DUTY = 100.0  # 手動で前進・後退するときのPWM（%）。
STEER_GAIN = 10.0  # 目標の横ずれが最大のとき、片側に足し引きするPWMのポイント数。大きいほど強く曲がる。
FIXED_AVOID_STEER_GAIN = 35.0  # 円柱・コーンへ寄る前に方向を変えるための操舵ゲイン（PWMのポイント数）。
FIXED_AVOID_TIGHT_BOTTOM = 0.78  # 物体の下端が画像高の78%以上で、まだ操舵が必要なら小回りへ切り替える。
FIXED_AVOID_INNER_DUTY_SCALE = 0.35  # 接近時に曲がる内側の履帯のPWM上限。SLOW_DUTYに対する倍率。
MAX_DUTY = 100.0  # 左右それぞれの出力上限（%）。手動時の出力にも適用される。
LEFT_TRIM = 0.0  # 自律で目標を追うときの左モータ補正。正なら左PWMを増やす（ポイント）。
RIGHT_TRIM = 0.0  # 自律で目標を追うときの右モータ補正。正なら右PWMを増やす（ポイント）。

# 画像の有効期限、待機・復帰・通過の設定。
# 画像の座標は左端・上端を0、右端・下端を1とする。物体の高さや面積も画像全体に対する割合で扱う。
IMAGE_MAX_AGE = 0.5  # 画像取得から判断までの許容時間（秒）。最終進入の更新間隔・赤の消失判定にも使う。
ENEMY_HOLD_SECONDS = 1.0  # 進路を塞いでいた相手を見失った場合に停止を保持する時間（秒）。
ENEMY_STOP_HALF_WIDTH_X = 0.12  # 予定進路の左右に取る停止判定の幅。車体の分も見込み、画像幅の12%ずつ。
ENEMY_AVOID_RANGE_X = 0.15  # 停止範囲の外側で低速回避を始める追加幅。画像幅に対する割合。
ENEMY_AVOID_OFFSET_X = 0.08  # 回避で予定目標をずらす最大量。大きくすると画像上の舵の補正が強くなる。
ENEMY_AVOID_DUTY_SCALE = 0.85  # 停止範囲のすぐ外での基本PWM倍率。1.0なら追加減速なし。
ENEMY_MIN_BOTTOM = 0.45  # 黄色の下端が画像の縦45%より上なら除外。試走2の画面上部の誤検出対策。
ENEMY_MIN_HEIGHT = 0.06  # 相手機体を判断に使う目印の高さ（画像高の6%）。
ENEMY_MIN_AREA = 0.004  # 高さが小さくても判断に使う目印の面積（画像全体の0.4%）。
ENEMY_CLOSE_HEIGHT = 0.23  # 目印がこの高さ以上なら至近距離とみなし、現在の車体正面も停止範囲に含める。
SEARCH_SECONDS = 5.0  # 目標を見失ったときに左右を探索する時間の上限（秒）。
STALL_SECONDS = 2.5  # 前進指令中に移動の証拠が得られず、復帰を試すまでの時間（秒）。
STATIC_CLEAR_HOLD_SECONDS = 0.2  # 目前の障害物が検出から外れてから、再発進までに空きを確認する時間（秒）。
MAX_RECOVERIES = 3  # 同じコース段階での復帰試行回数の上限。段階が変わるとリセット。
PASS_CLEAR_SECONDS = 0.4  # 側方の緑円柱を見失った後、通過判定までに必要な前進時間（秒）。
OBSTACLE_CLEARANCE_X = 0.09  # 発進時・手前の緑円柱の外縁から取る余裕。画像幅の9%。
FAR_OBSTACLE_CLEARANCE_X = 0.09  # 奥の緑円柱の外縁から取る余裕。画像幅の9%。
OBSTACLE_PASS_BOTTOM = 0.90  # 横に十分離れた円柱の下端がこの高さを超えたら、通過の確認を始める。
CONE_SIDE_MARGIN_X = 0.07  # コーンの外縁と進路中心の余裕。画像幅の7%。
CONE_CLEAR_MIN_BOTTOM = 0.82  # コーンの下端が画像高の82%以上に来てから、横を抜ける直進を許す。
CONE_CLEAR_MIN_HEIGHT = 0.22  # コーンの高さが画像高の22%以上であることも、接近の目安にする。
CONE_CLEAR_CONFIRM_FRAMES = 3  # 近さと横の余裕を連続何フレーム確認したら直進へ切り替えるか。
CONE_TRACK_MAX_JUMP_X = 0.18  # 追跡中に許す中心の横移動量。画像幅18%を超える青領域は別候補とする。
CONE_LOSS_SIDE_EDGE_X = 0.12  # コーンが横に見切れ始めたとみなす帯。左右の端から画像幅の12%。
CONE_LOSS_MIN_BOTTOM = 0.78  # 横端のコーンを近いとみなす下端の位置。画像高の78%以上。
CONE_LOSS_LOWER_EDGE = 0.90  # 横に十分離れたコーンが画像下端から見切れるとみなす位置。画像高の90%以上。
CONE_EDGE_TRACE_SIDE_X = 0.25  # 端の青を追跡中のコーンと結びつけるには、直前のコーンの外縁も同じ側の25%以内。
CONE_EDGE_TRACE_BOTTOM = 0.65  # 端の断片に移る前に、追跡中のコーンが画像高の65%以上まで近づいていること。
CONE_CLEAR_SECONDS = 1.0  # コーンが視野から消えた後に直進する最低時間（秒）。SLOW_DUTYで試走して調整する。

# ゴールの判定に使う面積は「赤い輪郭の面積 ÷ 処理画像全体の面積」。
GOAL_NEAR_AREA = 0.025  # 面積比2.5%。減速と、赤を見失う直前に十分近かったかの判定に使う。
GOAL_CREEP_AREA = 0.065  # 面積比6.5%。赤が中央付近にあれば微速進入に切り替える。
# ゴール調整画像1.pngでは台座の下端が画像高の約76%。ここでは黒い台座へ向かって前進を続ける。
GOAL_BASE_MATCH_X = 0.10  # 赤い目印の真下、または前の台座位置から許す横ずれ。画像幅の10%。
GOAL_BASE_MATCH_BOTTOM = 0.15  # 同じ台座として許す下端位置の変化。画像高の15%。
GOAL_BASE_FINAL_BOTTOM = 0.93  # 台座の下端が画像高の93%以上なら最終前進へ。大きくすると切り替えが遅くなる。
GOAL_BASE_FINAL_CENTER_X = 0.055  # 最終前進に必要な、台座中心と画像中央の最大ずれ。画像幅の5.5%。
GOAL_BASE_CONFIRM_FRAMES = 3  # 台座の近さと中央への整列を連続で確認する画像枚数。
GOAL_BASE_LOSS_SECONDS = 0.6  # 台座を途中で見失った場合、その場で再検出を待つ上限（秒）。
# 黒い台座に機体が重なる位置までの前進時間は、下のPWMで実機を走らせて測る。
GOAL_FINAL_ADVANCE_SECONDS = 3.0  # 最終前進時間（秒）。台座の手前で止まる場合は延ばす。Noneなら停止。
GOAL_FINAL_DUTY = 100.0  # 最後に直進するときの基本PWM（%）。前進時間はこのPWMで実測して調整する。
GOAL_FINAL_MAX_SECONDS = 5.0  # 視野外での前進時間に設定できる上限（秒）。
# 以前の試走では赤が見えるまま旧6秒の上限に達し、ゴールまで約60cm残った。
# 進む速さはDUTYとMOVE_SECONDS、台座へ近づく途中で時間切れになる場合はLIMITを調整する。
GOAL_CREEP_DUTY = 90.0  # 微速進入で前進するときの基本PWM（%）。旧80%から引き上げた。
GOAL_CREEP_LIMIT = 24.0  # 赤や台座を追う微速区間の上限（秒）。観測で停止している時間も含む。
GOAL_CREEP_PERIOD_SECONDS = 0.35  # 前進と停止を合わせた1周期の長さ（秒）。
GOAL_CREEP_MOVE_SECONDS = 0.30  # 各周期の前進時間（秒）。伸ばすとよく進む。停止時間を残すため周期未満にする。


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
        self.static_blocked = False
        self.static_clear_since = None
        self.enemies_in_range = []  # 現在の画像で停止・回避の判断に使える黄色い目印。
        self.track = None
        self.cone_track = None
        self.cone_seen_at = None  # 最後に追跡中のコーンを確認した時刻。
        self.cone_side_frames = 0
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
        self.visible_goal_base = None  # 今の画像で選んだ台座。相手の停止判定とゴールの操舵で共用する。
        self.goal_base_track = None  # 最後に選んだ黒い台座の位置と大きさ。
        self.goal_base_seen_at = -math.inf
        self.goal_base_confirmations = 0
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
        self.visible_goal_base = None
        self.goal_base_confirmations = 0
        # 取得異常の間を走行時間・通過実績として数えない。
        if self.track:
            self.track["edge_at"] = None
            self.track["clear_elapsed"] = None
        self.cone_side_frames = 0
        self.cone_clear_elapsed = None
        self.cone_seen_at = None
        self.static_clear_since = None
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

    def _drive_toward(self, x, duty, avoid_side=None, steer_gain=None):
        adjusted_x, adjusted_duty = self._avoid_enemy(x, duty, avoid_side)
        if (adjusted_x, adjusted_duty) != (x, duty):
            self.reason += "／相手を避けて予定側へ進行"
        x, duty = adjusted_x, adjusted_duty
        self.motion = stateMachine(self.motion, {"x": x}, None)
        error = (x - 0.5) * 2  # 中央からのずれを左端-1、右端+1に換算する。
        turn = (STEER_GAIN if steer_gain is None else steer_gain) * error
        base = duty * (1 - 0.35 * abs(error))  # ずれが大きいほど減速し、端では基本PWMを35%下げる。
        # 通常の追従では後退させず、前進の左右差で曲がる。
        left = min(MAX_DUTY, max(0, base + turn + LEFT_TRIM))
        right = min(MAX_DUTY, max(0, base - turn + RIGHT_TRIM))
        return left, right

    def _drive_around(self, obj, x, side):
        """固定障害物から離れる。近ければ旋回内側の履帯を抑えて前進量を減らす。"""
        left, right = self._drive_toward(x, SLOW_DUTY, avoid_side=side,
                                         steer_gain=FIXED_AVOID_STEER_GAIN)
        if x != 0.5 and obj["bottom"] >= FIXED_AVOID_TIGHT_BOTTOM:
            inner_limit = SLOW_DUTY * FIXED_AVOID_INNER_DUTY_SCALE
            if side == "left":
                left = min(left, inner_limit)
            else:
                right = min(right, inner_limit)
            self.reason += "／接近のため小回り"
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

    @staticmethod
    def _enemy_overlaps(enemy, x):
        if x is None:
            return True  # 探索や復帰の旋回では前進する進路を確定できない。
        left = right = x
        if enemy["height"] >= ENEMY_CLOSE_HEIGHT:
            # 目前では曲がり終わる前に車体正面が相手に達する可能性がある。
            left, right = min(x, 0.5), max(x, 0.5)
        return (enemy["right"] >= left - ENEMY_STOP_HALF_WIDTH_X and
                enemy["left"] <= right + ENEMY_STOP_HALF_WIDTH_X)

    def _enemy_blocks(self, scene, now):
        # 画面上部の黄色い物は避ける必要のある相手機体として数えない。
        visible = [enemy for enemy in scene["enemies"] if enemy["bottom"] >= ENEMY_MIN_BOTTOM]
        if visible:
            self.last_enemy = now  # 遠方に見えた相手も、後退の判断には使う。
        self.enemies_in_range = [enemy for enemy in visible if
                                 (enemy["height"] >= ENEMY_MIN_HEIGHT or
                                  enemy["area"] >= ENEMY_MIN_AREA)]
        if self.enemies_in_range:
            target = self._planned_target(scene, now)
            blocked = any(self._enemy_overlaps(enemy, target) for enemy in self.enemies_in_range)
            # 見失った場合だけ停止を保持。進路外に移ったと確認できればすぐに再開する。
            self.enemy_until = now + ENEMY_HOLD_SECONDS if blocked else -math.inf
        return now < self.enemy_until

    def _static_obstacle_blocks(self, scene, now):
        # 画面下端付近にある大きな物体だけ、直前の接触リスクとして扱う。
        obstacles = scene["obstacles"] + scene["cones"]
        if any(obj["bottom"] > 0.92 and obj["height"] > 0.2 and
               obj["left"] < 0.58 and obj["right"] > 0.42 for obj in obstacles):
            self.static_blocked = True
            self.static_clear_since = None
        elif self.static_blocked:
            if self.static_clear_since is None:
                self.static_clear_since = now
            if now - self.static_clear_since >= STATIC_CLEAR_HOLD_SECONDS:
                self.static_blocked = False
        return self.static_blocked

    def _avoid_enemy(self, x, duty, avoid_side):
        # コーン後部の通過待ちやゴールの最終前進では、直進方位と時間の校正を保つ。
        if not self.enemies_in_range or self.stage in ("CENTER_CLEAR", "GOAL_FINAL"):
            return x, duty
        # 停止範囲のすぐ外にいる相手のうち、予定進路に一番近いものを使う。
        enemy = min(self.enemies_in_range, key=lambda e: max(e["left"] - x, x - e["right"]))
        gap = max(enemy["left"] - x, x - enemy["right"])
        if gap <= ENEMY_STOP_HALF_WIDTH_X:
            return x, duty  # 進路と重なる相手はupdateの手前で停止する。
        distance = gap - ENEMY_STOP_HALF_WIDTH_X
        if distance >= ENEMY_AVOID_RANGE_X:
            return x, duty
        weight = 1 - distance / ENEMY_AVOID_RANGE_X
        duty *= 1 - (1 - ENEMY_AVOID_DUTY_SCALE) * weight
        direction = 1 if enemy["right"] < x else -1
        # 回避のための追加の舵は、コーンや円柱で既に選んだ側と同じ場合だけ加える。
        side = avoid_side
        if self.stage in ("CENTER_APPROACH", "CENTER_PASS"):
            side = self.route
        elif self.track:
            side = self.track["side"]
        if ((side == "left" and direction > 0) or (side == "right" and direction < 0) or
                (x - 0.5) * direction < 0):
            return x, duty
        offset = min(ENEMY_AVOID_OFFSET_X * weight, x if direction < 0 else 1 - x)
        adjusted = x + direction * offset
        if any(self._enemy_overlaps(other, adjusted) for other in self.enemies_in_range):
            adjusted = x  # 別の相手へ近づく補正は取り消す。
        return adjusted, duty

    @staticmethod
    def _nearest(objects):
        return max(objects, key=lambda obj: obj["bottom"] + obj["height"],
                   default=None)

    def _planned_target(self, scene, now):
        """相手がいないときに、この画像で走行側が使う目標の横座標を求める。"""
        if self.recovery:
            return None
        if self.stage == "GOAL_FINAL":
            return 0.5
        if self.visible_goal_base is not None:
            return self.visible_goal_base["x"]
        if self.stage == "GOAL_BASE":
            return None  # 台座を見失った間は停止して次の画像を待つ。
        if self.stage in ("CENTER_APPROACH", "CENTER_PASS", "CENTER_CLEAR"):
            cone = self._select_cone(scene)
            if cone is not None:
                return self._cone_target(cone)
            if self.stage == "CENTER_CLEAR":
                return 0.5  # コーンが見切れた後の直進。
            if self.stage == "CENTER_PASS":
                if self._can_follow_side_exit(scene, now):
                    return 0.5
                return None  # コーンを見失ったため、進路を決め直せない。
        if self.stage in ("START_EXIT", "NEAR_OBSTACLE", "FAR_OBSTACLE"):
            obj = self._tracked_obstacle(scene["obstacles"])
            if obj is not None:
                return self._obstacle_target(obj, self._obstacle_side(obj), self._obstacle_clearance())
            if self.track:
                return 0.5  # 円柱が側方に見切れた後の前進。
        flag = scene["flag"]
        return flag["x"] if flag else None

    def _tracked_obstacle(self, objects):
        if self.track is None:
            return self._nearest(objects)
        previous = self.track["object"]
        # 全候補を前回の円柱と照合する。別の円柱が最も大きくなっても追跡を捨てない。
        matches = [obj for obj in objects if abs(obj["x"] - previous["x"]) <= 0.3
                   and obj["height"] >= previous["height"] * 0.5]
        return min(matches, key=lambda obj: abs(obj["x"] - previous["x"]), default=None)

    def _obstacle_side(self, obj):
        if self.track:
            return self.track["side"]
        # 奥の円柱は、中央コーンで指定された通行側の反対を通るとゴール側へ抜けられる。
        if self.stage == "FAR_OBSTACLE" and self.route in ("left", "right"):
            return "left" if self.route == "right" else "right"
        return "left" if obj["x"] >= 0.5 else "right"

    def _obstacle_clearance(self):
        return FAR_OBSTACLE_CLEARANCE_X if self.stage == "FAR_OBSTACLE" else OBSTACLE_CLEARANCE_X

    @staticmethod
    def _obstacle_target(obj, side, clearance):
        # 円柱と余裕を含む範囲から、画面中央（現在の進行方向）を外すための最小限の舵。
        # すでに中央が空いていれば、左右どちらに回避中でも0.5に戻して直進する。
        if side == "left":
            return max(0.08, min(0.5, obj["left"] - clearance))
        return min(0.92, max(0.5, obj["right"] + clearance))

    def _pass_object(self, objects, next_stage, now, dt):
        """接近・側方への移動・前進の証拠をそろえて通過を推定する。"""
        obj = self._tracked_obstacle(objects)
        if self.track is None and obj is not None:
            self.track = {"object": obj, "side": self._obstacle_side(obj), "near": False,
                          "edge_at": None, "clear_elapsed": None, "evidence_at_loss": 0}
        track = self.track
        if track is None:
            return None
        x = 0.5  # 対象がいったん消えた場合は、それまでの方位を保つ。
        if obj is not None:
            track["object"] = obj
            track["clear_elapsed"] = None
            # 物体の下端が画像の縦65%より下で、高さが画像の16%を超えたら接近済みと記録。
            track["near"] |= obj["bottom"] > 0.65 and obj["height"] > 0.16
            # 左を通ると物体は右端の22%に入り、右を通ると左端の22%に入る。
            at_edge = (obj["x"] > 0.78 if track["side"] == "left"
                       else obj["x"] < 0.22)
            x = self._obstacle_target(obj, track["side"], self._obstacle_clearance())
            # 横に余裕がある円柱は、画面下端から見切れることもある。
            at_edge = x == 0.5 and (at_edge or obj["bottom"] > OBSTACLE_PASS_BOTTOM)
            if x != 0.5:
                track["edge_at"] = None  # 円柱が進路に戻ったら通過の確認をやり直す。
            if track["near"] and at_edge and track["edge_at"] is None:
                track["edge_at"] = now
        elif track["edge_at"] is not None:
            # まだ円柱が画面にあったときの前進は、後部が抜ける時間に含めない。
            if track["clear_elapsed"] is None or dt > IMAGE_MAX_AGE:
                track["clear_elapsed"] = 0.0
                track["evidence_at_loss"] = self.forward_evidence
            elif min(self.command) > 0:
                track["clear_elapsed"] += dt
            if (next_stage is not None and track["clear_elapsed"] >= PASS_CLEAR_SECONDS and
                    self.forward_evidence - track["evidence_at_loss"] >= 2):
                self._change_stage(next_stage, now)
                return (0.0, 0.0)  # 段階変更時は一度止め、次の画像で進路を確認。
        if (obj is None and track["edge_at"] is None and
                (track["near"] or self._obstacle_target(track["object"], track["side"],
                                                        self._obstacle_clearance()) != 0.5)):
            self.reason = "接近対象を前方で喪失。通過と決めつけず停止"
            return (0.0, 0.0)
        self.reason = "緑円柱を回避"
        if obj is not None:
            return self._drive_around(obj, x, track["side"])
        return self._drive_toward(x, SLOW_DUTY)

    def _select_cone(self, scene):
        candidates = scene["cones"]
        previous = self.cone_track
        if previous is not None:
            # 隣の服や椅子の青へ、追跡対象が急に切り替わるのを防ぐ。
            candidates = [obj for obj in candidates
                          if abs(obj["x"] - previous["x"]) <= CONE_TRACK_MAX_JUMP_X]
        if not candidates:
            return None
        flag = scene["flag"]
        reference = previous["x"] if previous is not None else flag["x"] if flag else 0.5
        return min(candidates, key=lambda obj: abs(obj["x"] - reference))

    def _cone_target(self, cone):
        # 予定方向の判定と実際の操舵で、同じコーンの回避目標を使う。
        if self.route == "left":
            return max(0.08, min(0.5, cone["left"] - CONE_SIDE_MARGIN_X))
        return min(0.92, max(0.5, cone["right"] + CONE_SIDE_MARGIN_X))

    @staticmethod
    def _cone_ahead(scene):
        # 追跡対象から外れた青領域でも、近くの正面にある場合は見切れとは判断しない。
        return any(obj["bottom"] >= CONE_LOSS_MIN_BOTTOM and
                   obj["height"] >= CONE_CLEAR_MIN_HEIGHT and
                   obj["left"] < 0.5 + CONE_SIDE_MARGIN_X and
                   obj["right"] > 0.5 - CONE_SIDE_MARGIN_X for obj in scene["cones"])

    def _has_tracked_cone_edge(self, scene):
        cone = self.cone_track
        if (cone is None or self.cone_seen_at is None or
                cone["bottom"] < CONE_EDGE_TRACE_BOTTOM or self._cone_target(cone) != 0.5):
            return False
        side = "right" if self.route == "left" else "left"  # コーンは選んだ通行側の反対に見える。
        if side not in scene.get("cone_edges", ()):
            return False
        if side == "left":
            return cone["left"] <= CONE_EDGE_TRACE_SIDE_X
        return cone["right"] >= 1 - CONE_EDGE_TRACE_SIDE_X

    def _can_follow_side_exit(self, scene, now):
        cone = self.cone_track
        if cone is None or self.cone_seen_at is None or self._cone_ahead(scene):
            return False
        if self._has_tracked_cone_edge(scene):
            # 今の画像にも同じ側にコーンの断片がある。停止して待っていた場合も再開できる。
            return True
        if (now - self.cone_seen_at > IMAGE_MAX_AGE or min(self.command) <= 0 or
                self._cone_target(cone) != 0.5):
            return False  # 断片すら見えない場合は、移動中の直近の観測だけを使う。
        if self.route == "left":
            at_side = cone["right"] >= 1 - CONE_LOSS_SIDE_EDGE_X
        else:
            at_side = cone["left"] <= CONE_LOSS_SIDE_EDGE_X
        # 下端で切れたコーンは高さが短く映るので、十分低い位置なら高さで弾かない。
        return (cone["bottom"] >= CONE_LOSS_LOWER_EDGE or
                (at_side and cone["bottom"] >= CONE_LOSS_MIN_BOTTOM and
                 cone["height"] >= CONE_CLEAR_MIN_HEIGHT))

    def _pass_cone(self, scene, now, dt):
        """指定側へ進入し、コーンが視野外に出た後も後部が抜けるまで直進する。"""
        cone = self._select_cone(scene)
        if dt > IMAGE_MAX_AGE:
            self.cone_side_frames = 0
        if cone is not None:
            self.cone_track = cone
            self.cone_seen_at = now
            # コーンが再び見えたら、消失後の直進時間は最初から数える。
            self.cone_clear_elapsed = None
            x = self._cone_target(cone)
            clear_side = x == 0.5  # 指定側に、向きを変えずに進める余裕が取れている。
            if self.stage == "CENTER_CLEAR" and not clear_side:
                # まだコーンが見えているので、奥へ向き直さず指定側の回避をやり直す。
                self._change_stage("CENTER_PASS", now)
                self.cone_side_frames = 0
                self.reason = "コーンとの余裕を取り直すため指定側へ修正"
                return self._drive_around(cone, x, self.route)
            if self.stage != "CENTER_CLEAR":
                near = (cone["bottom"] >= CONE_CLEAR_MIN_BOTTOM and
                        cone["height"] >= CONE_CLEAR_MIN_HEIGHT)
                self.cone_side_frames = self.cone_side_frames + 1 if near and clear_side else 0
                if self.cone_side_frames < CONE_CLEAR_CONFIRM_FRAMES:
                    self.stage = "CENTER_PASS"
                    self.reason = "指定側でコーンに進入"
                    return self._drive_around(cone, x, self.route)
                self._change_stage("CENTER_CLEAR", now)
        elif self.stage == "CENTER_APPROACH":
            return None  # まだコーンを見ていない間は、通常の追従で接近する。
        else:
            if self.stage != "CENTER_CLEAR":
                self.cone_side_frames = 0
                if not self._can_follow_side_exit(scene, now):
                    self.reason = "コーンの側方を確認できず停止。再検出を待つ"
                    return 0.0, 0.0
                # 側方が確認できたまま見切れた。3フレームに達していなくても、ここから後部の通過を確認する。
                self._change_stage("CENTER_CLEAR", now)
                self.cone_clear_elapsed = None
            if self._cone_ahead(scene):
                self.cone_clear_elapsed = None
                self.reason = "正面に青い物体があるため停止。コーンの通過確認を保留"
                return 0.0, 0.0
            if self._has_tracked_cone_edge(scene):
                self.cone_clear_elapsed = None
                self.reason = "側端にコーンの断片を確認。方位を保って直進"
                return self._drive_toward(0.5, SLOW_DUTY)
            if self.cone_clear_elapsed is None or dt > IMAGE_MAX_AGE:
                # 消失を最初に観測した時点を起点にする。それ以前の時間は含めない。
                self.cone_clear_elapsed = 0.0
                self.cone_evidence_at_loss = self.forward_evidence
            elif min(self.command) > 0:
                self.cone_clear_elapsed += dt  # 直前の指令が前進の場合だけ時間を足す。
            if (self.cone_clear_elapsed >= CONE_CLEAR_SECONDS and
                    self.forward_evidence - self.cone_evidence_at_loss >= 2):
                self._change_stage("FAR_OBSTACLE", now)
                self.cone_track = None
                self.cone_seen_at = None
                return 0.0, 0.0  # 次の画像から奥の障害物に向けて操舵する。

        if cone is not None:
            self.reason = "コーンの横を直進。車体後部が抜けるまで進路を維持"
        else:
            self.reason = "コーンが視野外。後部が抜けるまで直進"
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
        self.cone_track = None
        self.cone_seen_at = None
        self.cone_clear_elapsed = None
        self.cone_side_frames = 0
        self.last_progress = now
        return 0.0, 0.0

    def _select_goal_base(self, scene, now):
        """赤の真下で台座候補を探し、一度選んだ後は台座の位置を基準に追う。"""
        previous = self.goal_base_track
        if previous is None:
            if (now - self.last_goal_seen > IMAGE_MAX_AGE or
                    (self.last_goal_area < GOAL_NEAR_AREA and self.stage != "GOAL_CREEP")):
                return None
            reference_x = self.last_goal_x
        else:
            reference_x = previous["x"]
        candidates = [
            obj for obj in scene.get("goal_bases", ())
            if abs(obj["x"] - reference_x) <= GOAL_BASE_MATCH_X and
            (previous is None or
             abs(obj["bottom"] - previous["bottom"]) <= GOAL_BASE_MATCH_BOTTOM)
        ]
        return min(candidates, key=lambda obj: abs(obj["x"] - reference_x), default=None)

    def _start_final_goal(self, now):
        self._change_stage("GOAL_FINAL", now)
        self.final_driven_seconds = 0.0
        if GOAL_FINAL_ADVANCE_SECONDS is not None:
            self.final_deadline = now + GOAL_FINAL_ADVANCE_SECONDS
        return self._final_goal()

    def _final_goal(self):
        duration = GOAL_FINAL_ADVANCE_SECONDS
        if duration is None or not (0 < duration <= GOAL_FINAL_MAX_SECONDS):
            self.stop("最終前進時間が未校正のため停止")
            return self.command
        if self.final_driven_seconds >= duration:
            if self.goal_base_track is not None:
                self.stop("最終前進を完了。黒い台座への到達は推定")
                self.phase = "GOAL_HOLD"
            else:
                self.stop("最終前進を完了。黒い台座への到達は未確認")
            return self.command
        self.reason = ("黒い台座に正対。設定時間だけ最終前進" if self.goal_base_track is not None
                       else "赤が視野外。設定時間だけ最終前進")
        return self._drive_toward(0.5, GOAL_FINAL_DUTY)

    def _goal(self, flag, now):
        if self.stage == "GOAL_FINAL":
            # 画面上端で赤が映ったり消えたりしても、最終前進の方位と期限を保つ。
            return self._final_goal()
        if self.creep_started is not None and now - self.creep_started >= GOAL_CREEP_LIMIT:
            self.stop("ゴールの微速進入上限。台座への到達は未確認")
            return self.command
        base = self.visible_goal_base
        if base is not None:
            if self.stage != "GOAL_BASE":
                self._change_stage("GOAL_BASE", now)
                if self.creep_started is None:
                    self.creep_started = now
            if now - self.goal_base_seen_at > IMAGE_MAX_AGE:
                self.goal_base_confirmations = 0
            if (base["bottom"] >= GOAL_BASE_FINAL_BOTTOM and
                    abs(base["x"] - 0.5) <= GOAL_BASE_FINAL_CENTER_X):
                self.goal_base_confirmations += 1
            else:
                self.goal_base_confirmations = 0
            self.goal_base_track = base
            self.goal_base_seen_at = now
            if self.goal_base_confirmations >= GOAL_BASE_CONFIRM_FRAMES:
                return self._start_final_goal(now)
            target_x = base["x"]
            self.reason = "黒い台座の中央へ微速進入"
        elif self.stage == "GOAL_BASE":
            self.goal_base_confirmations = 0
            # 台座の位置を確認してから進む。再検出できるまでは車体を動かさない。
            if now - self.goal_base_seen_at >= GOAL_BASE_LOSS_SECONDS:
                self.stop("黒い台座を再検出できず停止。到達未確認")
            else:
                self.reason = "黒い台座を再検出するまで停止"
            return 0.0, 0.0
        elif flag is None:
            # 近接と整列を3回以上確認し、最後の赤が中央から7%幅未満で、観測も新しければ最終進入へ。
            near_and_aligned = (self.near_goal_confirmations >= 3 and
                                self.last_goal_area >= GOAL_NEAR_AREA and
                                abs(self.last_goal_x - 0.5) < 0.07 and
                                now - self.last_goal_seen <= IMAGE_MAX_AGE)
            if near_and_aligned:
                # 台座を追えない場合は従来の短い前進を使い、台座への到達は未確認として扱う。
                return self._start_final_goal(now)
            if self.stage == "GOAL_CREEP" or self.last_goal_area >= GOAL_NEAR_AREA:
                self.stop("ゴール付近の消失を確認できないため停止")
                return self.command
            return self._search(now)
        else:
            target_x = flag["x"]
            error = abs(target_x - 0.5)
            # 中央から7%幅未満で近接面積も満たすフレームを数える。外れたら数え直す。
            if flag["area"] >= GOAL_NEAR_AREA and error < 0.07:
                self.near_goal_confirmations += 1
            else:
                self.near_goal_confirmations = 0
            if self.stage == "GOAL_ALIGN":
                if error < 0.05:  # 最初の整列は中央から5%幅未満で完了。
                    self._change_stage("GOAL_APPROACH", now)
                return self._drive_toward(target_x, SLOW_DUTY)
            if self.stage == "GOAL_APPROACH":
                if flag["area"] >= GOAL_CREEP_AREA and error < 0.07:
                    self._change_stage("GOAL_CREEP", now)
                    self.creep_started = now
                else:
                    duty = SLOW_DUTY if flag["area"] >= GOAL_NEAR_AREA else DRIVE_DUTY
                    return self._drive_toward(target_x, duty)
            self.reason = "ゴール微速進入（到達は画像による推定）"
        elapsed = now - self.creep_started
        # 各周期の先頭で前進し、残りの時間は停止して観測する。
        if elapsed % GOAL_CREEP_PERIOD_SECONDS >= GOAL_CREEP_MOVE_SECONDS:
            return 0.0, 0.0
        return self._drive_toward(target_x, GOAL_CREEP_DUTY)

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
            self.cone_side_frames = 0
            self.cone_track = None
            self.cone_seen_at = None
            if self.stage == "CENTER_CLEAR":
                # 手動の間に姿勢が変わり得るので、再開時はコーンの側方を確かめ直す。
                self.stage = "CENTER_PASS"
                self.cone_clear_elapsed = None
            if self.stage.startswith("GOAL_"):
                # 手動の前に見えた目印や台座の位置を、再開後の進路に使わない。
                self._change_stage("GOAL_ALIGN", now)
                self.near_goal_confirmations = 0
                self.last_goal_seen = -math.inf
                self.last_goal_area = 0.0
                self.last_goal_x = 0.5
                self.creep_started = None
                self.goal_base_track = None
                self.goal_base_seen_at = -math.inf
                self.goal_base_confirmations = 0
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
        self.visible_goal_base = None
        if self.stage in ("GOAL_ALIGN", "GOAL_APPROACH", "GOAL_CREEP", "GOAL_BASE"):
            self.visible_goal_base = self._select_goal_base(scene, now)
        if self._enemy_blocks(scene, now):
            if self.stage == "GOAL_FINAL":
                self.stop("最終進入中に相手を検出。到達未確認で停止")
                return self.command
            self.command = (0.0, 0.0)
            self.reason = "予定進路上の相手／通過待ち"
            self.last_progress = now  # 意図した待機をスタックと判定しない。
            self.near_goal_confirmations = 0
            self.goal_base_confirmations = 0
            # 相手による遮蔽中に通過時間だけを消費しない。
            if self.track:
                self.track["edge_at"] = None
                self.track["clear_elapsed"] = None
            self.cone_clear_elapsed = None
            self.cone_side_frames = 0
            self.static_clear_since = None
            return self.command
        if self._static_obstacle_blocks(scene, now):
            if self.stage == "GOAL_FINAL":
                self.stop("最終進入の前方が塞がったため停止")
                return self.command
            self.command = (0.0, 0.0)
            self.reason = "目前の障害物。接触を避けるため進路が空くまで停止"
            self.last_progress = now  # 意図的に止めている間は復帰のタイマーを進めない。
            self.recovery = None
            self.goal_base_confirmations = 0
            if self.track:
                self.track["edge_at"] = None
                self.track["clear_elapsed"] = None
            self.cone_clear_elapsed = None
            self.cone_side_frames = 0
            return self.command
        if self.recovery:
            self.command = self._recover(now)
            return self.command
        if (self.stage not in ("GOAL_CREEP", "GOAL_BASE", "GOAL_FINAL") and
                self.search_started is None and min(self.command) > 0 and
                now - self.last_progress >= STALL_SECONDS):
            if self.enemies_in_range:
                self.command = (0.0, 0.0)
                self.reason = "相手が近いため復帰旋回を待機"
            elif self.stage == "CENTER_CLEAR":
                self.stop("コーンの横で前進を確認できないため停止")
            else:
                self._begin_recovery(scene, now)
            return self.command
        self.search_started = None if flag else self.search_started
        if self.stage == "START_EXIT":
            # 発進から0.4秒以上経ち、前進の証拠が2回以上あれば次の段階へ進む。
            if self.forward_evidence >= 2 and now - self.stage_entered >= 0.4:
                track = self.track
                self._change_stage("NEAR_OBSTACLE", now)
                self.track = track  # 発進時に選んだ円柱と回避側をそのまま引き継ぐ。
        if self.stage == "START_EXIT":
            command = self._pass_object(scene["obstacles"], None, now, dt)
        elif self.stage in ("NEAR_OBSTACLE", "FAR_OBSTACLE"):
            next_stage = "CENTER_APPROACH" if self.stage == "NEAR_OBSTACLE" else "GOAL_ALIGN"
            command = self._pass_object(scene["obstacles"], next_stage, now, dt)
        elif self.stage in ("CENTER_APPROACH", "CENTER_PASS", "CENTER_CLEAR"):
            command = self._pass_cone(scene, now, dt)
        elif self.stage.startswith("GOAL_"):
            command = self._goal(flag, now)
        else:
            command = None
        if command is None:
            if flag:
                command = self._drive_toward(flag["x"], DRIVE_DUTY)
            else:
                command = self._search(now)
        self.command = command
        return self.command

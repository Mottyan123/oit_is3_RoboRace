# ライブラリ等のインポート ------------------------------------------
import cv2
import numpy as np
import locateTarget as lt
import stateMachine as sm
import threading
import time

from ClsDualMotorControl import ClsDualMotorControl
#from ClsDualMotorControlDummy import ClsDualMotorControl

# イメージセンサの初期化 ------------------------------------------
videoCap = cv2.VideoCapture(0)
sWidthSensor = 640
sHeightSensor = 480
videoCap.set(cv2.CAP_PROP_FRAME_WIDTH, sWidthSensor)
videoCap.set(cv2.CAP_PROP_FRAME_HEIGHT, sHeightSensor)

# 処理画像サイズの決定 ------------------------------------------
sResizeRatio = 0.5
if videoCap.isOpened():
	sReturn, imCamera = videoCap.read()
	print('W: ', imCamera.shape[1], ', H:', imCamera.shape[0])
	sWidth = int(imCamera.shape[1] * sResizeRatio)
	sHeight = int(imCamera.shape[0] * sResizeRatio)

# 画像処理・表示ループ外の変数 ------------------------------------
sDisplayRate = 5
sMode = 1
sFrame = 0

# 画像記録バッファ -----------------------------------------------
imGray = np.ndarray((sHeight, sWidth))
imRedBinary = np.ndarray((sHeight, sWidth))
imGreenBinary = np.ndarray((sHeight, sWidth))

# ステート初期化 -------------------------------------------------
sState = sm.IDLE

# Dual motor controller初期設定 ---------------------------------
vPortsDrive = [23, 22, 25, 9, 10]  #AIN1, AIN2, BIN1, BIN2, STBY
vPortsPWM = [12, 13]
sFrequency = 400
ClsDmc = ClsDualMotorControl(vPortsDrive, vPortsPWM, sFrequency)

# 画像処理・表示ループ -------------------------------------------------
class MotorOutput:
    """GPIO出力の窓口を1つにし、画像取得が止まっても期限で停止する。"""

    def __init__(self, driver):
        self.driver = driver
        self.lock = threading.Lock()
        self.closed = threading.Event()
        self.lease_expired = threading.Event()
        self.deadline = None
        self.valid_until = time.monotonic() + 1.0
        self.directions = None
        self.error = None
        self.worker = threading.Thread(target=self._watch, daemon=True)
        self.worker.start()

    def _stop_locked(self):
        self.driver.stop()
        self.directions = None

    def _watch(self):
        # waitは監視スレッドだけで行う。画像処理やキー入力を待たせない。
        while not self.closed.wait(0.05):
            try:
                with self.lock:
                    now = time.monotonic()
                    if self.deadline is not None and now >= self.deadline:
                        self._stop_locked()
                    elif now >= self.valid_until:
                        self.lease_expired.set()
                        self._stop_locked()
            except Exception as error:
                self.error = str(error)
                self.closed.set()

    def submit(self, command, deadline):
        with self.lock:
            now = time.monotonic()
            self.deadline = deadline
            self.valid_until = now + 1.0  # 指令更新が1秒途切れたら停止。
            if (self.error or (deadline is not None and now >= deadline)
                    or not all(np.isfinite(value) for value in command)):
                self._stop_locked()
                return
            # 制御側は(左,右)、ドライバはモータ0＝右、1＝左。
            duties = (command[1], command[0])
            directions = tuple(0 if value > 0 else 1 if value < 0 else None
                               for value in duties)
            if directions != self.directions:
                # 既存driveMotorは反対側ピンを下げないので、反転前に必ずstop。
                self._stop_locked()
                for number, duty in enumerate(duties):
                    if duty:
                        self.driver.driveMotor(number, directions[number],
                                               min(sm.MAX_DUTY, abs(duty)))
                self.directions = directions
            else:
                # 同方向の速度変更はPWMだけを更新し、不要な停止を減らす。
                for number, duty in enumerate(duties):
                    self.driver.setPWM(min(sm.MAX_DUTY, abs(duty)), number)

    def close(self):
        self.closed.set()
        self.worker.join(timeout=0.2)
        with self.lock:
            self._stop_locked()


controller = sm.RaceController()
motor_output = MotorOutput(ClsDmc)
previous_message = None

try:
    while videoCap.isOpened():
        now = time.monotonic()
        if motor_output.lease_expired.is_set():
            controller.sensor_failure(now)
            motor_output.lease_expired.clear()
        if motor_output.error:
            raise RuntimeError("モータ出力の監視に失敗: " + motor_output.error)

        # キーはモードごとに解釈する。e・qとモード変更は画像取得前にも停止。
        sKey = cv2.waitKey(1) & 0xFF
        old_mode = controller.mode
        controller.handle_key(sKey, now)
        sMode = controller.mode
        if (old_mode != sMode or sKey in (ord('e'), ord('q'))
                or controller.check_deadline(time.monotonic())):
            motor_output.submit((0.0, 0.0),
                                controller.deadline if sMode == 2 else None)
        if controller.quit:
            break

        sReturn, imCamera = videoCap.read()
        received_at = time.monotonic()
        if motor_output.lease_expired.is_set():
            # read中に監視停止した場合、古い通過・移動の証拠を使わない。
            controller.sensor_failure(received_at)
            motor_output.lease_expired.clear()
        if (not sReturn or imCamera is None or imCamera.ndim != 3
                or imCamera.shape[2] != 3):
            controller.sensor_failure(received_at)
            motor_output.submit((0.0, 0.0),
                                controller.deadline if sMode == 2 else None)
            continue

        # 実際に取得できた寸法を使う。中心160等を固定値にしない。
        sWidth = max(1, int(imCamera.shape[1] * sResizeRatio))
        sHeight = max(1, int(imCamera.shape[0] * sResizeRatio))
        imResize = cv2.resize(imCamera, (sWidth, sHeight))
        scene = lt.observe(imResize)
        scene['received_at'] = received_at
        command = controller.update(scene, time.monotonic())
        # 出力直前にも期限を確認する。監視スレッドとの排他もsubmitが担当。
        # 手動モードには3分制限を適用しない（更新途絶時の停止は共通）。
        output_deadline = controller.deadline if sMode == 2 else None
        if sMode == 2 and controller.stage == 'GOAL_FINAL' and min(command) > 0:
            # 最後の前進も監視スレッドで打ち切り、次の画像を待って走り過ぎない。
            if controller.final_deadline is not None:
                output_deadline = min(output_deadline, controller.final_deadline)
        motor_output.submit(command, output_deadline)
        sState = controller.motion

        imDisplay = imResize.copy()
        groups = [(scene['flag'], (0, 0, 255), 'goal')]
        groups += [(obj, (0, 255, 0), 'obstacle') for obj in scene['obstacles']]
        groups += [(obj, (255, 0, 0), 'cone') for obj in scene['cones']]
        groups += [(obj, (0, 255, 255), 'robot') for obj in scene['enemies']]
        for obj, color, label in groups:
            if obj is not None:
                x, y, w, h = obj['bbox']
                cv2.rectangle(imDisplay, (x, y), (x + w, y + h), color, 1)
                cv2.putText(imDisplay, label, (x, max(12, y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        elapsed = 0 if controller.started_at is None else time.monotonic() - controller.started_at
        status = '{} {} {} {} {:.1f}s'.format(
            sMode, controller.phase, controller.stage, controller.route or '-', elapsed)
        cv2.putText(imDisplay, status, (3, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, (255, 255, 255), 1)
        goal_area = scene['flag']['area'] if scene['flag'] else 0.0
        details = 'L:{:.0f} R:{:.0f} goal_area:{:.4f}'.format(
            command[0], command[1], goal_area)
        cv2.putText(imDisplay, details, (3, 29), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (255, 255, 255), 1)
        message = (controller.mode, controller.phase, controller.stage, controller.reason)
        if message != previous_message:
            print(status, controller.reason)
            previous_message = message
        if sFrame % sDisplayRate == 0:
            cv2.imshow('input', imDisplay)
        sFrame = (sFrame + 1) % 1000

except KeyboardInterrupt:
    controller.stop('割込みで停止')

except Exception:
    controller.stop('例外で停止')
    raise

# 終了処理 ----------------------------------------------------
finally:
    # 通常終了だけでなく、認識処理等の例外でも両輪を止める。
    try:
        motor_output.close()
    finally:
        videoCap.release()
        cv2.destroyAllWindows()

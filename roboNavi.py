"""カメラ画像とキー入力を受け取り、自律・手動走行の指令をモータへ渡す。"""

import threading
import time

import cv2
import numpy as np

import locateTarget as lt
import stateMachine as sm

# カメラと表示の設定。認識には実際のカメラ画像を縦横とも半分に縮めて使う。
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
PROCESS_SCALE = 0.5
DISPLAY_INTERVAL_FRAMES = 5  # 画像処理は毎回行い、ウィンドウは5フレームごとに更新する。

# モータ0は右側、モータ1は左側。ピン番号はGPIOの番号で指定する。
DRIVE_PINS = (23, 22, 25, 9, 10)  # AIN1, AIN2, BIN1, BIN2, STBY
PWM_PINS = (12, 13)
PWM_FREQUENCY_HZ = 400

WATCH_INTERVAL_SECONDS = 0.05  # 停止期限を確認する周期。
COMMAND_TIMEOUT_SECONDS = 1.0  # この時間だけ指令の更新がなければ止める。


class MotorOutput:
    """カメラの処理が止まっても、別スレッドでモータの停止期限を監視する。"""

    def __init__(self, driver):
        self.driver = driver
        # メインループの出力と監視スレッドの停止が重ならないようにする。
        self.lock = threading.Lock()
        self.closed = threading.Event()
        self.lease_expired = threading.Event()
        self.deadline = None
        self.valid_until = time.monotonic() + COMMAND_TIMEOUT_SECONDS
        self.directions = None
        self.error = None
        self.worker = threading.Thread(target=self._watch, daemon=True)
        self.worker.start()

    def _stop_locked(self):
        """GPIOを操作する前に、呼び出し側でlockを取っておく。"""
        self.driver.stop()
        self.directions = None

    def _watch(self):
        # waitは監視スレッド内で行うので、画像処理や入力を待たせない。
        while not self.closed.wait(WATCH_INTERVAL_SECONDS):
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
        """左右のPWM指令を適用する。負のPWMは後退を表す。"""
        with self.lock:
            now = time.monotonic()
            self.deadline = deadline
            self.valid_until = now + COMMAND_TIMEOUT_SECONDS
            deadline_reached = deadline is not None and now >= deadline
            if self.error or deadline_reached or not all(np.isfinite(command)):
                self._stop_locked()
                return

            # 制御側の指令は(左, 右)。ドライバは右をモータ0に割り当てている。
            left, right = command
            duties = (right, left)
            directions = tuple(
                0 if duty > 0 else 1 if duty < 0 else None for duty in duties
            )
            if directions != self.directions:
                # 反転前にstopで両方の方向ピンを落とす。
                self._stop_locked()
                for number, duty in enumerate(duties):
                    if duty:
                        self.driver.driveMotor(
                            number, directions[number], min(sm.MAX_DUTY, abs(duty))
                        )
                self.directions = directions
            else:
                # 方向が同じなら、PWMの変更だけで速度を調整できる。
                for number, duty in enumerate(duties):
                    self.driver.setPWM(min(sm.MAX_DUTY, abs(duty)), number)

    def close(self):
        """監視を終了し、最後に両モータを停止する。"""
        self.closed.set()
        self.worker.join(timeout=0.2)
        with self.lock:
            self._stop_locked()


def _valid_camera_frame(captured, frame):
    """カメラが取得できたBGR画像かを確認する。"""
    return captured and frame is not None and frame.ndim == 3 and frame.shape[2] == 3


def _draw_overlay(frame, scene, status, command):
    """表示用のコピーに認識結果と走行状態を書き込む。"""
    display = frame.copy()
    groups = [
        ([scene["flag"]], (0, 0, 255), "goal"),
        (scene["obstacles"], (0, 255, 0), "obstacle"),
        (scene["cones"], (255, 0, 0), "cone"),
        (scene["enemies"], (0, 255, 255), "robot"),
    ]
    for objects, color, label in groups:
        for obj in objects:
            if obj is None:
                continue
            x, y, width, height = obj["bbox"]
            cv2.rectangle(display, (x, y), (x + width, y + height), color, 1)
            cv2.putText(
                display, label, (x, max(12, y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1,
            )

    cv2.putText(
        display, status, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1
    )
    goal_area = scene["flag"]["area"] if scene["flag"] else 0.0
    left, right = command
    details = f"L:{left:.0f} R:{right:.0f} goal_area:{goal_area:.4f}"
    cv2.putText(
        display, details, (3, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1
    )
    return display


def run_navigation(camera, driver):
    """キー入力、画像認識、走行判断、モータ出力の順で処理する。"""
    controller = sm.RaceController()
    motor_output = MotorOutput(driver)
    previous_message = None
    frame_number = 0

    try:
        while camera.isOpened():
            now = time.monotonic()
            if motor_output.lease_expired.is_set():
                controller.sensor_failure(now)
                motor_output.lease_expired.clear()
            if motor_output.error:
                raise RuntimeError("モータ出力の監視に失敗: " + motor_output.error)

            # 停止キーとモード変更は、次の画像の取得を待たずモータに反映する。
            key = cv2.waitKey(1) & 0xFF
            old_mode = controller.mode
            controller.handle_key(key, now)
            mode = controller.mode
            # 3分制限は自律のみ。指令更新の途絶による停止は手動でも有効。
            race_deadline = controller.deadline if mode == 2 else None
            stop_requested = old_mode != mode or key in (ord("e"), ord("q"))
            if stop_requested or controller.check_deadline(time.monotonic()):
                motor_output.submit((0.0, 0.0), race_deadline)
            if controller.quit:
                break

            captured, frame = camera.read()
            received_at = time.monotonic()
            if motor_output.lease_expired.is_set():
                # 撮影を待つ間に監視側で停止したら、以前の通過・移動の情報を破棄する。
                controller.sensor_failure(received_at)
                motor_output.lease_expired.clear()
            if not _valid_camera_frame(captured, frame):
                controller.sensor_failure(received_at)
                motor_output.submit((0.0, 0.0), race_deadline)
                continue

            height, width = frame.shape[:2]
            processed_width = max(1, int(width * PROCESS_SCALE))
            processed_height = max(1, int(height * PROCESS_SCALE))
            processed = cv2.resize(frame, (processed_width, processed_height))
            scene = lt.observe(processed)
            scene["received_at"] = received_at
            command = controller.update(scene, time.monotonic())

            # updateで自律開始が確定する場合があるので、出力時点で期限を取り直す。
            output_deadline = controller.deadline if mode == 2 else None
            if (
                mode == 2
                and controller.stage == "GOAL_FINAL"
                and controller.final_deadline is not None
                and min(command) > 0
            ):
                # 次の画像が届かなくても、最終前進だけは設定時間で打ち切る。
                output_deadline = min(output_deadline, controller.final_deadline)
            motor_output.submit(command, output_deadline)

            elapsed = 0.0
            if controller.started_at is not None:
                elapsed = time.monotonic() - controller.started_at
            route = controller.route or "-"
            status = f"{mode} {controller.phase} {controller.stage} {route} {elapsed:.1f}s"
            display = _draw_overlay(processed, scene, status, command)
            message = (mode, controller.phase, controller.stage, controller.reason)
            if message != previous_message:
                print(status, controller.reason)
                previous_message = message
            if frame_number % DISPLAY_INTERVAL_FRAMES == 0:
                cv2.imshow("input", display)
            frame_number = (frame_number + 1) % 1000

    except KeyboardInterrupt:
        controller.stop("割込みで停止")
    except Exception:
        controller.stop("例外で停止")
        raise
    finally:
        # カメラや認識で例外が起きても、両モータを止めてから抜ける。
        motor_output.close()


def main():
    """実機のカメラとGPIOを準備し、走行ループを起動する。"""
    from ClsDualMotorControl import ClsDualMotorControl
    # PCで試す場合は、必要に応じて上の import をダミーに切り替える。
    # from ClsDualMotorControlDummy import ClsDualMotorControl

    camera = cv2.VideoCapture(CAMERA_INDEX)
    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        if camera.isOpened():
            captured, frame = camera.read()
            if _valid_camera_frame(captured, frame):
                height, width = frame.shape[:2]
                print("W: ", width, ", H:", height)

        driver = ClsDualMotorControl(DRIVE_PINS, PWM_PINS, PWM_FREQUENCY_HZ)
        run_navigation(camera, driver)
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

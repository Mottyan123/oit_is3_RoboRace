"""左右のモータを GPIO と PWM で動かす。"""

import lgpio


class ClsDualMotorControl:
    """方向ピンとPWMピンを操作する2モータ用ドライバ。"""

    def __init__(self, vPortsDrive, vPortsPWM, sFrequency):
        """方向ピンは [AIN1, AIN2, BIN1, BIN2, STBY] の順で渡す。"""
        self.__hGPIO = lgpio.gpiochip_open(0)
        self.__vPortsDrive = vPortsDrive
        self.resetPort()

        self.__vPortsPWM = vPortsPWM  # [モータ0のPWMピン, モータ1のPWMピン]
        self.__sFrequency = sFrequency  # PWMの周波数（Hz）。
        self.setPWM(0, 0)

    def __del__(self):
        # GPIOを閉じる前に、方向信号と両モータのPWMを落とす。
        self.stop()
        lgpio.gpiochip_close(self.__hGPIO)

    def resetPort(self):
        """4本の方向ピンとSTBYを0にし、ドライバを待機状態にする。"""
        for pin_index in range(5):
            pin = self.__vPortsDrive[pin_index]
            lgpio.gpio_write(self.__hGPIO, pin, 0)

    def setPWM(self, sDuty, sMotorNumber):
        """指定したモータの出力をデューティ比（0～100%）で設定する。"""
        pin = self.__vPortsPWM[sMotorNumber]
        lgpio.tx_pwm(self.__hGPIO, pin, self.__sFrequency, sDuty)

    def stop(self):
        """ドライバを待機状態にし、左右の出力を0%にする。"""
        self.resetPort()
        self.setPWM(0, 0)
        self.setPWM(0, 1)

    def driveMotor(self, sMotorNumber, sDirection, sDuty):
        """モータ番号と方向（どちらも0または1）を選んで駆動する。"""
        # モータ0はAIN1/AIN2、モータ1はBIN1/BIN2を使う。
        direction_index = sMotorNumber * 2 + sDirection
        standby_pin = self.__vPortsDrive[4]
        direction_pin = self.__vPortsDrive[direction_index]

        # STBYを1にすると出力できる。呼び出し側は反転前にstop()を呼ぶ。
        lgpio.gpio_write(self.__hGPIO, standby_pin, 1)
        lgpio.gpio_write(self.__hGPIO, direction_pin, 1)
        self.setPWM(sDuty, sMotorNumber)


if __name__ == "__main__":
    import time

    # このファイルを直接実行したときのモータ試験。
    drive_pins = [23, 22, 25, 9, 10]
    pwm_pins = [12, 13]
    frequency_hz = 10000

    driver = ClsDualMotorControl(drive_pins, pwm_pins, frequency_hz)
    driver.stop()

    try:
        while True:
            for motor_number, direction in ((0, 0), (0, 1), (1, 0), (1, 1)):
                driver.driveMotor(motor_number, direction, 80)
                time.sleep(2)
    except KeyboardInterrupt:
        del driver

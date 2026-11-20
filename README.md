# oit_is3_RoboRace

情報科学演習Ⅲ テーマ2のロボット競技用プログラムです。Raspberry Piの前方カメラでコース上の目印を認識し、左右のモータを制御します。手動操作と、通行側を指定して開始する自律走行の2つのモードがあります。

## 実行環境と起動

- Raspberry Pi、前方カメラ、左右のモータとモータドライバ
- Python 3、OpenCV（`cv2`）、NumPy、`lgpio`
- OpenCVの表示ウィンドウを開ける環境。SSHで操作する場合はX11転送（`ssh -YC`）を使用できます。

ワークスペースのルートで実行し、表示された `input` ウィンドウを選択してキーを入力します。

```sh
python3 roboNavi.py
```

カメラ番号は `0`、要求解像度は640 × 480です。画像認識には、取得した画像を縦横とも半分に縮めた画像を使います。GPIOの配線とPWM周波数は [モータ制御の仕様書](docs/ClsDualMotorControl/ClsDualMotorControl_specification.pdf) に記載しています。

## キー操作

| キー | 手動モード | 自律モード |
| --- | --- | --- |
| `1` | 手動モードに切り替える | 手動モードに切り替える |
| `2` | 自律モードに切り替える | 自律モードを維持する |
| `w` | 前進 | |
| `s` | 後退 | 通行側を選択した後に走行開始 |
| `a` / `d` | 左回転 / 右回転 | |
| `l` / `r` | | 車体の進行方向を基準に、コーンの左側 / 右側を選択 |
| `e` | 停止 | 停止。走行開始後はその回の自律走行を終了 |
| `q` | 停止してプログラムを終了 | 停止してプログラムを終了 |

起動時は手動モードです。自律走行は **`2` → `l` または `r` → `s`** の順で開始します。制限時間は自律走行に適用され、開始キーから179.8秒で停止するように設定されています。手動モードでも、画像が無効なときやモータへの指令が途絶えたときは停止します。

## 自律走行の流れ

1. 発進し、手前の緑の円柱を避けます。
2. 青い中央コーンの指定側を通り、車体後部が抜けるまで直進します。
3. 奥の緑の円柱は、コーンの指定側と反対側を通ってゴール側に抜けます。
4. ゴールポールの赤いひし形に近づき、黒い台座が見えたら台座の中央を追います。台座に正対して十分近づいたことを確認すると、設定時間だけ直進して停止します。

黄色い目印の相手が予定進路を塞いでいる間は停止します。進路のすぐ外にいる相手には、区間に応じて減速や操舵で対応します。競技上のゴールは機体が黒い台座に重なった位置です。プログラムはカメラ画像と最終前進の時間から到達を推定します。

## 調整とログ

端末ログには、モード、走行状態、コース段階、通行側、開始からの経過時間、判断理由が表示されます。`CENTER_CLEAR` はコーン後部の通過確認、`GOAL_BASE` は黒い台座の追跡、`GOAL_FINAL` は最後の時間指定の直進です。

| 調整したい内容 | 設定 |
| --- | --- |
| 赤・緑・青・黄の認識 | [`locateTarget.py`](locateTarget.py) の `HSV_RANGES` |
| 黒い台座の認識 | [`locateTarget.py`](locateTarget.py) の `GOAL_BASE_*` |
| 円柱・コーンとの間隔 | [`stateMachine.py`](stateMachine.py) の `OBSTACLE_CLEARANCE_X`、`FAR_OBSTACLE_CLEARANCE_X`、`CONE_SIDE_MARGIN_X` |
| 相手との停止・回避距離 | [`stateMachine.py`](stateMachine.py) の `ENEMY_*` |
| ゴールの最終前進 | [`stateMachine.py`](stateMachine.py) の `GOAL_BASE_FINAL_BOTTOM`、`GOAL_FINAL_ADVANCE_SECONDS`（現在は3.0秒） |

画像内の位置や間隔に使う比率は、画面の左端・上端を0、右端・下端を1として表します。たとえば横方向の `0.09` は画像幅の9%です。

## スクリプトと資料

| スクリプト | 役割 | フローチャート | 仕様書 |
| --- | --- | --- | --- |
| [`roboNavi.py`](roboNavi.py) | カメラ、キー入力、表示、モータ出力の監視 | [PDF](docs/roboNavi/roboNavi_flowchart.pdf) | [PDF](docs/roboNavi/roboNavi_specification.pdf) |
| [`locateTarget.py`](locateTarget.py) | 目印の色・形・位置を画像から検出 | [PDF](docs/locateTarget/locateTarget_flowchart.pdf) | [PDF](docs/locateTarget/locateTarget_specification.pdf) |
| [`stateMachine.py`](stateMachine.py) | 走行段階の管理、回避・停止の判断、左右のPWM計算 | [PDF](docs/stateMachine/stateMachine_flowchart.pdf) | [PDF](docs/stateMachine/stateMachine_specification.pdf) |
| [`ClsDualMotorControl.py`](ClsDualMotorControl.py) | GPIOで方向・待機信号とPWMを操作 | [PDF](docs/ClsDualMotorControl/ClsDualMotorControl_flowchart.pdf) | [PDF](docs/ClsDualMotorControl/ClsDualMotorControl_specification.pdf) |

4本のスクリプトの処理図と説明を、進捗レポート用紙の表裏にまとめています。 [進捗レポートの記入例（A4・2ページ）](docs/report/roborace_progress_report.pdf)

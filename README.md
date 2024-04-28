 
### ✨ はじめに
今回のプロジェクトは、短大二年の理系女子がテンソルフローを自作 PC で試すため YouTube の動画を参考にして作ったよ！

とりあえずエラーなしで動くとこまで（笑）

今回ボクが参考にした動画（下のほうにリンクある）では、Google Colaboratory を使ってるけど、ボクは自作 ㍶ でローカルの環境を使いたっかたので、色々ググりながらやりました。

てな訳で、ボクの環境を例として説明するとこんな感じです。

>- CPU: *Ryzen ７ 7800 X3D* 
>- Python 3.6.x
>- Tensorflow 2.6.2
>- pip 21.3.1
>- OS は Windows 11 で、エディタはマイクロソフトの VSコード 使ってるで。

TensorFlowのインストール手順:

最初に、Pythonのパッケージ管理システムであるpipを最新の状態にアップデートするんやで。
```bash
pip install --upgrade pip
```
次に、TensorFlowをインストールするんやで。
```bash
pip install tensorflow
```

AMD Ryzen 系 CPU 向けの TensorFlow は、`tensorflow` ではなく `tensorflow-rocm` パッケージを使うんやで。
```bash
pip install tensorflow-rocm
```
インストールが完了したら、テンソルフローが正しくインストールされたかどうかを確認するために、以下のコードを実行してみるんやで。

他のヴァージョンのテンソルフローを探してる場合はこっちみて。=> 
[ソースからビルドする < Tensorflow](https://www.tensorflow.org/install/source?hl=ja#gpu)

### ✏️ 参考にした動画   
『Python TensorFlow for Machine Learning – Neural Network Text Classification Tutorial』by Kylie Ying (カイル ヤング) さん
[https://youtu.be/VtRLrQ3Ev-U?si=l8jnaQ6DIdXQbcXr](https://youtu.be/VtRLrQ3Ev-U?si=l8jnaQ6DIdXQbcXr)

![動画](/images/gifit_1714286853814.gif)


#### ✍️ コースの概要:
このコースは、PythonとTensorFlowを使って機械学習の基本的なコンセプトやニューラルネットワークの実装について学ぶことができるで。Kylie Ying (カイル ヤング) さんが基本的な概念、例えば分類、回帰、トレーニング/検証/テストデータセット、損失関数、ニューラルネットワーク、モデルのトレーニングなどを説明してくれるで。その後、feedforwardニューラルネットワークを実装して、誰かが糖尿病を持っているかどうかを予測する方法や、ワインのレビューを分類するための異なるニューラルネットワークアーキテクチャをデモしてくれるで。


#### ⭐️ 動画に上げてあるリソース ⭐️
💻 Datasets: [https://drive.google.com/drive/folder...](https://drive.google.com/drive/folders/1YnxDqNIqM2Xr1Dlgv5pYsE6dYJ9MGxcM)
💻 Feedforward NN colab notebook: [https://colab.research.google.com/dri...](https://drive.google.com/drive/folders/1YnxDqNIqM2Xr1Dlgv5pYsE6dYJ9MGxcM)
💻 Wine review colab notebook: [https://colab.research.google.com/dri...](https://colab.research.google.com/drive/1yO7EgCYSN3KW8hzDTz809nzNmacjBBXX?usp=sharing)


### 🍧 おまけ
ちなみにかしこいボクは、スタンフォードと MIT 大学院生がつくった高校生向けのコンピューターサイエンス課外講習の Inspirit AI でアンバサダーしてるので、夏期講習に興味ある方は ↓ の QR コードをスキャンしてな。えっと英語でだからそこんとこちゃんとできるようにしてや！

![Inspirit AI Summer 2024 Invite](/images/inspiritai-su24-ai.jpg)

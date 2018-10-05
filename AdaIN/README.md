# AdaIN

## Summary
![here](https://github.com/SerialLain3170/Style-Transfer/blob/master/AdaIN/images/network.png)
- Encoderを通した後、AdaINで変換してDecoderで復元
- AdaINではスタイル画像のチャンネル毎の平均と標準偏差を用いて変換。これによりどんなスタイル画像を与えても変換可能に。
- AdaINの式は以下

![here](https://github.com/SerialLain3170/Style-Transfer/blob/master/AdaIN/images/adain.png)  
xがcontent画像、yがstyle画像である。

## Usage
```bash
$ python train.py --alpha <ALPHA>
```
で実行。ALPHAは0<=α<=1で、値が大きいほどStyle画像要素が強くなる。

## Result
私の環境で生成した画像を以下に示す。

![here](https://github.com/SerialLain3170/Style-Transfer/blob/master/AdaIN/images/real.png)
![here](https://github.com/SerialLain3170/Style-Transfer/blob/master/AdaIN/images/anime.png)

まだcontent要素が強い。ハイパーパラメータ等詳細は以下に示す。
- 画像サイズは256×256
- バッチサイズは3
- 最適化手法はAdam(α=0.0001, β1=0.9)
- Style lossの計算にはVGG16のConv1_1、Conv2_1、Conv3_1、Conv4_1出力を考慮。Style lossの重みは10.0とした

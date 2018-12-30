# HairStyleTransfer

## Summary
![here](https://github.com/SerialLain3170/Style-Transfer/blob/master/HairStyleTransfer/network.png)

- InstaGANを用いて髪型をインスタンスとして見た髪型変換を行う
- 髪型だけのSegmentation maskを用意する

## Result
私の環境で生成した画像を以下に示す。
![here](https://github.com/SerialLain3170/Style-Transfer/blob/master/HairStyleTransfer/result.png)

まだ出来としては不十分ではあるが、方向性は間違ってないように見える

- バッチサイズは3
- 最適化手法はAdam(α=0.0002, β1=0.5)
- Data AugmentationとしてはHorizontal Flipを用いた
- Adversarial loss, Cycle-consistency loss, Identity-mapping loss, Context preserving lossの重みはそれぞれ

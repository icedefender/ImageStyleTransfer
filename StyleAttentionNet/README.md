# StyleAttentionNetwork

## Summary
![network](https://github.com/SerialLain3170/ImageStyleTransfer/blob/master/StyleAttentionNet/network.png)

- 元論文は[こちら](https://arxiv.org/pdf/1812.02342.pdf)
- VGGで抽出した特徴量に対してStyle Attentionを用いている
- 実装では論文中の２つのIdentity lossの重みは逆である。[参考](https://github.com/dypark86/SANET/issues/1)

## Result
私の環境で生成した画像例を示す
![result](https://github.com/SerialLain3170/ImageStyleTransfer/blob/master/StyleAttentionNet/transfer_result.png)

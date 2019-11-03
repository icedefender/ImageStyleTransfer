# RelGAN
## Summary
- 従来のMany-to-ManyなStarGANとは違い、変換前と変換先の補間も行える
- 補間具合を表すαを予測するようなDiscriminatorを導入している

# Result
私の環境で生成した例を以下に示す

![RelGAN](https://github.com/SerialLain3170/ImageStyleTransfer/blob/master/RelGAN/RelGAN_result.jpg)

- バッチサイズは16
- 最適化手法はAdam(α=0.00005, β1=0.9)
- interpolation loss, cycle-consistency loss, self-reconstruction lossの重みはそれぞれ10.0
- 論文中で用いられているOrthogonal regularizationを用いていない、Zero-centered gradient penaltyを用いている

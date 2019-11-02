# U-GAT-IT
## Summary
- CycleGANの枠組みにおいてClass Activation Map(CAM)を利用してスタイル変換を行う
- 論文中の結果を見る限りCycleGANよりも変換対象の特徴抽出は出来ている

## Results
私の環境で生成した例を以下に示す
![result](https://github.com/SerialLain3170/ImageStyleTransfer/blob/master/UGATIT/Result.jpg)

- バッチサイズは16
- 最適化手法はAdam(α=0.0001, β1=0.5, β2=0.999)
- Cycle-consistency loss, Identity-mapping loss, CAM lossの重みはそれぞれ10.0, 10.0, 1000.0

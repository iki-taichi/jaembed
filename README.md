# jaembed
JaEmbedID - A Japanese Character Embedding for Chainer

Chainer用の学習済み日本語文字埋め込み

学習済み(固定)トークンと学習可能トークンをハイブリットで使用可能


## 説明

JaEmbedIDは、[例文集合を入力とするニューラルネットワークを用いた文のスタイル変換に向けて](https://jsai-slud.github.io/sig-slud/)において使用した日本語文字埋め込みの実装と事前学習モデルです。

文字埋め込みのためモデル全体の容量は小さくしつつ、ベクトルに変換可能な文字種は広く取ることができます。

ASCIIコード(0~127)、JIS第1水準、JIS 第2水準、JIS非漢字の合計7666文字を対象として、日本語版wikipediaの全記事ダンプを用いて学習し、出現しな
かった文字を除外した7341文字に関する文字埋め込みベクトルを提供します。

文字埋め込みベクトルの学習には[gensimライブラリのWord2Vec](https://radimrehurek.com/gensim/models/word2vec.html)を用い、ベクト
ル次元128、窓幅8、10エポックの条件のもと学習しています。


## 環境

chainerに依存します。

Version 4.2.0 での動作を確認しています。


## 使い方

dataディレクトリとjaembed.pyを使用環境にコピーし、Chainerのlinkとして他のchainから呼び出します。

sample.pyでは、文字列の埋め込みをベクトルを取得するサンプルを示します。

```:python
import chainer
from jaembed import JaEmbedID

class SampleModel(chainer.Chain):
    def __init__(
            self, 
            metadata_path=None, 
            embed_path=None, 
            special_tokens=None
        ):
        super(SampleModel, self).__init__()
        
        self.charvec_dim  = 128
        
        HeN = chainer.initializers.HeNormal
        with self.init_scope():
            self.embed = JaEmbedID(
                    embed_path, 
                    special_tokens, 
                    metadata_path,
                    initialW=HeN(fan_option='fan_out')
                )
```

### 初期化時


```:python
# 学習済みベクトルデータのパスと学習させたいトークンを指定
model = SampleModel(
        embed_path='data/sg_d128_w8_mc0_neg5_iter10_s0p001.pklb',
        special_tokens=['<BOS>', '<EOS>', '<UNK>']
    )

# 辞書データの保存
model.embed.store_metadata('sample_dir/embed_meta.pklb')

# モデル保存
model.to_cpu()
serializers.save_npz('sample_dir/sample.model', model)
```

### モデルロード時

```:python
# メタデータを指定
model = SampleModel(
        metadata_path='sample_dir/embed_meta.pklb',
    )
model_path = os.path.join('sample_dir/sample.model', model_name)
serializers.load_npz('sample_dir/sample.model', model)
```

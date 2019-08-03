import torch
import torch.nn as nn

def preprocess(x):
    # unsqueezeによって次元を (4) -> (1,4)
    x = torch.tensor(x, dtype=torch.float).unsqueeze(0)
    return x

class QFunc(nn.Module):
    def __init__(self, obs_size=4, acs_size=2):
        super(QFunc, self).__init__()
        self.obs_size = obs_size
        self.acs_size = acs_size

        # 2層の全結合層を定義
        self.l1 = nn.Linear(in_features=obs_size, out_features=50)
        self.l2 = nn.Linear(in_features=50, out_features=acs_size)

    # 順伝播
    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = self.l2(x)
        return x
    
    # 行動選択用
    def select_action(self, x):
        x = preprocess(x)

        # no_gradの間は計算グラフを構築しない
        with torch.no_grad():
            q = self.forward(x)
        # 2次元目の最大値を取得してnumpyに変換
        action = q.max(1)[1][0].numpy()
        return action

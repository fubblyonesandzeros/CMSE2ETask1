import torch
from torch import nn


class ChannelWiseSelfAttention(nn.Module):
    def __init__(self, num_heads, channels, embed_dim):
        super().__init__()

        self.num_heads = num_heads
        self.channels = channels
        self.embed_dim = embed_dim

        self.q = nn.Linear(channels, embed_dim)
        self.k = nn.Linear(channels, embed_dim)
        self.v = nn.Linear(channels, embed_dim)

        self.out_proj = nn.Linear(embed_dim, channels)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x of shape (N, C, H, W)
        # x -> (N, H*W, C)

        batch_size = x.shape[0]
        num_channels = x.shape[1]
        height = x.shape[2]
        width = x.shape[3]

        x = x.view(batch_size, num_channels, -1)
        x = x.permute(0, 2, 1) # (N, H*W, C)

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = q.view(batch_size, q.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.view(batch_size, k.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.view(batch_size, v.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)

        scores = q @ k.transpose(-2, -1)
        scores /= q.shape[-1] ** .5

        scores = self.softmax(scores)

        out = scores @ v

        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, -1 ,self.embed_dim)
        out = self.out_proj(out)

        out = out.view(batch_size, height, width, num_channels).permute(0, 3, 1, 2)

        return out


class SpatialWiseSelfAttention(nn.Module):
    def __init__(self, num_heads, height, width, embed_dim):
        super().__init__()

        self.num_heads = num_heads
        self.features_dim = height * width
        self.embed_dim = embed_dim

        self.q = nn.Linear(self.features_dim, embed_dim)
        self.k = nn.Linear(self.features_dim, embed_dim)
        self.v = nn.Linear(self.features_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, self.features_dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x of shape (N, C, H, C)
        # x -> (N, C, H*W)

        batch_size = x.shape[0]
        num_channels = x.shape[1]
        height = x.shape[2]
        width = x.shape[3]

        x = x.view(batch_size, num_channels, -1)

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = q.view(batch_size, q.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.view(batch_size, k.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.view(batch_size, v.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)

        scores = q @ k.transpose(-2, -1)
        scores /= q.shape[-1] ** .5

        scores = self.softmax(scores)

        out = scores @ v

        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, -1 ,self.embed_dim)
        out = self.out_proj(out)

        out = out.view(batch_size, num_channels, height, width)

        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout

        self.channels_proj = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0
        )

        self.block = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels, 
                out_channels=out_channels, 
                kernel_size=3, 
                stride=1, 
                padding=1
            ),
            nn.Dropout2d(dropout),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels, 
                out_channels=out_channels, 
                kernel_size=3,
                stride=1, 
                padding=1
            ),
            nn.Dropout2d(dropout)
        )

    def forward(self, x):
        x = self.channels_proj(x)
        x = x + self.block(x)
        return x
        

class ResNet(nn.Module):
    def __init__(self, in_channels=2, starting_channels=16, depth=3, dropout=.2):
        super().__init__()

        self.in_channels = in_channels
        self.starting_channels = starting_channels
        self.dropout = dropout
        self.depth = depth

        blocks = []

        for depth in range(depth):
            blocks.append(
                ResBlock(
                    in_channels=in_channels if depth == 0 else starting_channels * depth,
                    out_channels=starting_channels * (depth + 1),
                    dropout=dropout
                )
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.blocks(x)
        x = x.mean(axis=(2, 3))
        return x
    

class ResNetWithChannelSpatialAttention(nn.Module):
    def __init__(
            self, 
            in_channels=2, 
            starting_channels=16, 
            depth=3, 
            img_height=32,
            img_width=32,
            num_heads=4, 
            embed_dim=32, 
            dropout=.2):
        super().__init__()

        self.in_channels = in_channels
        self.starting_channels = starting_channels
        self.dropout = dropout
        self.depth = depth
        self.img_height = img_height
        self.img_width = img_width
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        blocks = []

        for depth in range(depth):
            blocks.append(
                nn.Sequential(
                    ResBlock(
                        in_channels=in_channels if depth == 0 else starting_channels * depth,
                        out_channels=starting_channels * (depth + 1),
                        dropout=dropout
                    ),
                    ChannelWiseSelfAttention(
                        num_heads=num_heads, 
                        channels=starting_channels * (depth + 1), 
                        embed_dim=embed_dim
                    ),
                    SpatialWiseSelfAttention(
                        num_heads=num_heads,
                        height=img_height,
                        width=img_width,
                        embed_dim=embed_dim,
                    )
                )
                
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.blocks(x)
        x = x.mean(axis=(2, 3))
        return x


class Classifier(nn.Module):
    def __init__(self, encoder, hidden_size, num_classes):
        super().__init__()
        
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    x = torch.zeros((1, 2, 32, 32))


    classifier = Classifier(
        ResNetWithChannelSpatialAttention(
            in_channels=2, 
            img_height=32, 
            img_width=32
        ),
        hidden_size=48,
        num_classes=3
    )

    p = classifier(x)

    print(p.shape)
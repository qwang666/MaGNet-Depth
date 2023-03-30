import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def _round_to_multiple_of(val, divisor, round_up_bias=0.9):
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def _get_depths(alpha):
    """ Scales tensor depths as in reference MobileNet code, prefers rouding up
    rather than down. """
    depths = [32, 16, 24, 40, 80, 96, 192, 320]
    return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]


def agg_node(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
    )

def upshuffle(in_planes, out_planes, upscale_factor):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes*upscale_factor**2, kernel_size=3, stride=1, padding=1),
        nn.PixelShuffle(upscale_factor),
        nn.ReLU()
    )


def smooth(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
    )


def predict(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=1),
        nn.ReLU(),
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1),
        nn.Sigmoid(),
    )


class Encoder(nn.Module):
    def __init__(self, alpha=1.0):
        super(Encoder, self).__init__()
        
        if alpha == 1.0:
            MNASNet = torchvision.models.mnasnet1_0(
                weights=torchvision.models.MNASNet1_0_Weights.DEFAULT, progress=True)
            # MNASNet = torchvision.models.mnasnet1_0(
            #     pretrained=True, progress=True)
        else:
            MNASNet = torchvision.models.MNASNet(alpha=alpha)
            
        self.conv0 = nn.Sequential(
            MNASNet.layers._modules['0'],
            MNASNet.layers._modules['1'],
            MNASNet.layers._modules['2'],
            MNASNet.layers._modules['3'],
            MNASNet.layers._modules['4'],
            MNASNet.layers._modules['5'],
            MNASNet.layers._modules['6'],
            MNASNet.layers._modules['7'],
            # MNASNet.layers._modules['8'],
        )
        self.conv1 = MNASNet.layers._modules['8']
        self.conv2 = MNASNet.layers._modules['9']
        self.conv3 = MNASNet.layers._modules['10']

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        features = [x]
        features.append(conv0)
        features.append(conv1)
        features.append(conv2)
        features.append(conv3)
        
        return features


# Upsample depth via learned upsampling
def upsample_depth_via_mask(depth, up_mask, downsample_ratio):
    # depth: low-resolution depth (B, 2, H, W)
    # up_mask: (B, 9*k*k, H, W)
    k = downsample_ratio

    N, o_dim, H, W = depth.shape
    up_mask = up_mask.view(N, 1, 9, k, k, H, W)
    up_mask = torch.softmax(up_mask, dim=2)             # (B, 1, 9, k, k, H, W)

    up_depth = F.unfold(depth, [3, 3], padding=1)       # (B, 2, H, W) -> (B, 2 X 3*3, H*W)
    up_depth = up_depth.view(N, o_dim, 9, 1, 1, H, W)   # (B, 2, 3*3, 1, 1, H, W)
    up_depth = torch.sum(up_mask * up_depth, dim=2)     # (B, 2, k, k, H, W)

    up_depth = up_depth.permute(0, 1, 4, 2, 5, 3)       # (B, 2, H, k, W, k)
    return up_depth.reshape(N, o_dim, k*H, k*W)         # (B, 2, kH, kW)


class Decoder(nn.Module):
    def __init__(self, alpha=1.0):
        super(Decoder, self).__init__()
        depths = _get_depths(alpha)
        final_chs = depths[4]
        # Aggregate layers
        self.agg1 = agg_node(final_chs, final_chs)
        self.agg2 = agg_node(final_chs, 1)
        self.agg3 = agg_node(final_chs, 1)
        
        self.inner1 = nn.Conv2d(depths[3], final_chs, 1, bias=True)
        self.inner2 = nn.Conv2d(depths[2], final_chs, 1, bias=True)
        
        self.depth1 = nn.Sequential(nn.Conv2d(depths[4]+1+depths[3], depths[4]+1+depths[3], kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(depths[4]+1+depths[3], final_chs, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),)
        self.depth2 = nn.Sequential(nn.Conv2d(depths[4]+1+depths[2], depths[4]+1+depths[2], kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(depths[4]+1+depths[2], final_chs, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),)
        self.depth3 = nn.Sequential(nn.Conv2d(depths[4]+depths[1], depths[4]+depths[1], kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(depths[4]+depths[1], final_chs, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),)
        
        # depth prediction 
        self.depth_head = nn.Sequential(
            nn.Conv2d(final_chs, depths[3], 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(depths[3], depths[3], 1), nn.ReLU(inplace=True),
            nn.Conv2d(depths[3], 2, 1),
        )
        
        self.mask_head = nn.Sequential(
                nn.Conv2d(final_chs, depths[3], 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(depths[3], depths[3], 1), nn.ReLU(inplace=True),
                nn.Conv2d(depths[3], 9 * 2 * 2, 1)
            )
        self.upsample_depth = upsample_depth_via_mask
        
        
    def forward(self, features):
        conv0, conv1, conv2, conv3 = features[1], features[2], features[3], features[4]
        
        intra_feat = conv3
        # out = self.out1(intra_feat)
        depth_1 = self.agg1(intra_feat)

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv2)

        # out = self.out2(intra_feat)
        depth_2 = self.agg2(intra_feat)

        depth = self.depth1(torch.cat([F.interpolate(depth_1, scale_factor=2, mode='bilinear', align_corners=True), depth_2, conv2], dim=1)) # channel 80+1+40 to 80
        
        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv1)

        # out = self.out3(intra_feat)

        depth_3 = self.agg3(intra_feat)

        depth = self.depth2(torch.cat([F.interpolate(depth, scale_factor=2, mode='bilinear', align_corners=True), depth_3, conv1], dim=1))  # channel 80+1+24 to 80
        
        depth = self.depth3(torch.cat([F.interpolate(depth, scale_factor=2, mode='bilinear', align_corners=True), conv0], dim=1))  # channel 80+16 to 80

        depth_pred = self.depth_head(depth)

        mask = self.mask_head(depth)
        up_depth = self.upsample_depth(depth_pred, mask, 2)
        
        return up_depth


class MnasMulti(nn.Module):

    def __init__(self, alpha=1.0):
        super(MnasMulti, self).__init__()
        self.encoder = Encoder(alpha)
        self.decoder = Decoder(alpha)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        return self.decoder.parameters()
    
# class MnasMulti(nn.Module):

#     def __init__(self, alpha=1.0):
#         super(MnasMulti, self).__init__()
#         depths = _get_depths(alpha)
#         if alpha == 1.0:
#             MNASNet = torchvision.models.mnasnet1_0(
#                 weights=torchvision.models.MNASNet1_0_Weights.DEFAULT, progress=True)
#             # MNASNet = torchvision.models.mnasnet1_0(
#             #     pretrained=True, progress=True)
#         else:
#             MNASNet = torchvision.models.MNASNet(alpha=alpha)

#         self.conv0 = nn.Sequential(
#             MNASNet.layers._modules['0'],
#             MNASNet.layers._modules['1'],
#             MNASNet.layers._modules['2'],
#             MNASNet.layers._modules['3'],
#             MNASNet.layers._modules['4'],
#             MNASNet.layers._modules['5'],
#             MNASNet.layers._modules['6'],
#             MNASNet.layers._modules['7'],
#             # MNASNet.layers._modules['8'],
#         )
#         self.conv1 = MNASNet.layers._modules['8']
#         self.conv2 = MNASNet.layers._modules['9']
#         self.conv3 = MNASNet.layers._modules['10']

#         self.out1 = nn.Conv2d(depths[4], depths[4], 1, bias=False)

#         final_chs = depths[4]
#         self.inner1 = nn.Conv2d(depths[3], final_chs, 1, bias=True)
#         self.inner2 = nn.Conv2d(depths[2], final_chs, 1, bias=True)

#         self.out2 = nn.Conv2d(final_chs, depths[3], 3, padding=1, bias=False)
#         self.out3 = nn.Conv2d(final_chs, depths[2], 3, padding=1, bias=False)

#         # self.depth1 = nn.Conv2d(depths[4], 1, 1, bias=True)
#         # self.depth2 = nn.Conv2d(depths[3], 1, 1, bias=True)
#         # self.depth3 = nn.Conv2d(depths[2], 1, 1, bias=True)
#         # self.depth_pred = nn.Conv2d(3, 1, 1, bias=True)

#         # Smooth layers
#         self.smooth1 = nn.Conv2d(depths[4], depths[4], kernel_size=3, stride=1, padding=1)
#         self.smooth2 = nn.Conv2d(depths[4], depths[4], kernel_size=3, stride=1, padding=1)

#         # Aggregate layers
#         self.agg1 = agg_node(final_chs, final_chs)
#         self.agg2 = agg_node(final_chs, 1)
#         self.agg3 = agg_node(final_chs, 1)

#         # Upshuffle layers
#         self.up1 = upshuffle(final_chs, final_chs, 4)
#         self.up2 = upshuffle(final_chs, final_chs, 2)

#         # Depth prediction
#         # self.predict1 = smooth(final_chs*4, final_chs*4)
#         self.predict = predict(final_chs, 1)
#         # self.predict3 = nn.Sequential(nn.Conv2d(depths[2], 1, kernel_size=1, stride=1),
#         #                             #   nn.Sigmoid(),
#         #                               )
        
#         self.depth1 = nn.Sequential(nn.Conv2d(depths[4]+1+depths[3], depths[4]+1+depths[3], kernel_size=3, stride=1, padding=1),
#                                     nn.ReLU(),
#                                     nn.Conv2d(depths[4]+1+depths[3], final_chs, kernel_size=3, stride=1, padding=1),
#                                     nn.ReLU(),)
#         self.depth2 = nn.Sequential(nn.Conv2d(depths[4]+1+depths[2], depths[4]+1+depths[2], kernel_size=3, stride=1, padding=1),
#                                     nn.ReLU(),
#                                     nn.Conv2d(depths[4]+1+depths[2], final_chs, kernel_size=3, stride=1, padding=1),
#                                     nn.ReLU(),)
#         self.depth3 = nn.Sequential(nn.Conv2d(depths[4]+depths[1], depths[4]+depths[1], kernel_size=3, stride=1, padding=1),
#                                     nn.ReLU(),
#                                     nn.Conv2d(depths[4]+depths[1], final_chs, kernel_size=3, stride=1, padding=1),
#                                     nn.ReLU(),)
        
#         # depth prediction 
#         self.depth_head = nn.Sequential(
#             nn.Conv2d(final_chs, depths[3], 3, padding=1), nn.ReLU(inplace=True),
#             nn.Conv2d(depths[3], depths[3], 1), nn.ReLU(inplace=True),
#             nn.Conv2d(depths[3], 2, 1),
#         )
#         # self.decode_1 = nn.Sequential(nn.Conv2d(depths[4]+depths[3], depths[4]+depths[3], kernel_size=3, stride=1, padding=1),
#         #                               nn.ReLU(),
#         #                               nn.Conv2d(depths[4]+depths[3], depths[4], kernel_size=3, stride=1, padding=1),
#         #                               nn.ReLU(),)
#         # self.decode_2 = nn.Sequential(nn.Conv2d(depths[4]+depths[2], depths[4]+depths[2], kernel_size=3, stride=1, padding=1),
#         #                               nn.ReLU(),
#         #                               nn.Conv2d(depths[4]+depths[2], depths[4], kernel_size=3, stride=1, padding=1),
#         #                               nn.ReLU(),)
#         # self.depth_pred = nn.Sequential(
#         #     nn.Conv2d(depths[2], depths[2], 3, padding=1, bias=True),
#         #     # nn.BatchNorm2d(depths[2]),
#         #     nn.ReLU(inplace=True),

#         #     nn.Conv2d(depths[2], depths[2], 3, padding=1, bias=True),
#         #     # nn.BatchNorm2d(depths[2]),
#         #     nn.ReLU(inplace=True),

#         #     nn.Conv2d(depths[2], 1, 1, bias=False),
#         # )

#     def forward(self, x):
#         conv0 = self.conv0(x)
#         conv1 = self.conv1(conv0)
#         conv2 = self.conv2(conv1)
#         conv3 = self.conv3(conv2)
#         depth_scale = []
        
#         intra_feat = conv3
#         out = self.out1(intra_feat)
#         depth_1 = self.agg1(intra_feat)

#         intra_feat = F.interpolate(
#             intra_feat, scale_factor=2, mode="bilinear", align_corners=True) + self.inner1(conv2)

#         out = self.out2(intra_feat)
#         depth_2 = self.agg2(intra_feat)
#         depth_scale.append(F.interpolate(depth_2, size=(480, 640), mode='bilinear', align_corners=True))
        
#         depth = self.depth1(torch.cat([F.interpolate(depth_1, scale_factor=2, mode='bilinear', align_corners=True), depth_2, conv2], dim=1)) # channel 80+1+40 to 80
        
#         intra_feat = F.interpolate(
#             intra_feat, scale_factor=2, mode="bilinear", align_corners=True) + self.inner2(conv1)

#         out = self.out3(intra_feat)

#         depth_3 = self.agg3(intra_feat)
#         depth_scale.append(F.interpolate(depth_3, size=(480, 640), mode='bilinear', align_corners=True))
        
#         depth = self.depth2(torch.cat([F.interpolate(depth, scale_factor=2, mode='bilinear', align_corners=True), depth_3, conv1], dim=1))  # channel 80+1+24 to 80
        
#         depth = self.depth3(torch.cat([F.interpolate(depth, scale_factor=2, mode='bilinear', align_corners=True), conv0], dim=1))  # channel 80+16 to 80
        
        
#         # depth_1 = self.depth1(F.interpolate(depth_1, size=(H, W), mode='bilinear', align_corners=True))
#         # depth_2 = self.depth2(F.interpolate(depth_2, size=(H, W), mode='bilinear', align_corners=True))
#         # depth_3 = self.depth3(F.interpolate(depth_3, size=(H, W), mode='bilinear', align_corners=True))
#         # vol = torch.cat([depth_1, depth_2, depth_3], dim=1)
#         # vol = torch.cat([vol, F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)], dim=1)
#         depth_pred = self.depth_head(depth)
#         # depth_pred = self.depth_pred(torch.cat((F.interpolate(depth_1, scale_factor=4, mode="bilinear", align_corners=True),
#         #                                         F.interpolate(depth_2, scale_factor=2, mode="bilinear", align_corners=True),
#         #                                         depth_3), dim = 1))

#         return depth_pred

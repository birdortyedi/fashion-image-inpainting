import torch
from torch import nn


class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        var_w = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]).mean()
        var_h = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]).mean()

        return var_w + var_h


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.pixel = nn.L1Loss()
        self.content = nn.L1Loss()
        self.style = nn.L1Loss()
        self.adversarial = nn.BCELoss()
        self.tv = TVLoss()

    def forward(self, x, output, composite, d_out, vgg_features_gt, vgg_features_output, vgg_features_composite):
        pixel_whole_loss = self.pixel(x, output)
        pixel_composite_loss = self.pixel(x, composite)
        pixel_loss = pixel_whole_loss + pixel_composite_loss

        s_loss_output, s_loss_composite = 0.0, 0.0
        for i, (f_gt, f_composite, f_output) in enumerate(zip(vgg_features_gt, vgg_features_composite, vgg_features_output)):
            g_f_gt = self._gram_matrix(f_gt)
            g_f_output = self._gram_matrix(f_output)
            g_f_composite = self._gram_matrix(f_composite)
            s_loss_output += self.style(g_f_output, g_f_gt).mean()
            s_loss_composite += self.style(g_f_composite, g_f_gt).mean()

        style_loss = s_loss_output + s_loss_composite

        tv_loss = self.tv(composite)

        adversarial_loss = self.adversarial(d_out,
                                            torch.FloatTensor(d_out.size(0)).uniform_(0.0, 0.3).to(torch.device("cuda" if torch.cuda.is_available()
                                                                                                                else "cpu")))

        return 10.0 * pixel_loss + 120.0 * style_loss + 0.001 * adversarial_loss + 0.0001 * tv_loss, pixel_loss, style_loss, adversarial_loss, tv_loss

    @staticmethod
    def _gram_matrix(mat):
        b, ch, h, w = mat.size()
        m = mat.view(b, ch, w * h)
        m_transposed = m.transpose(1, 2)
        # G = m.bmm(m_transposed) / (h * w * ch)
        i_ = torch.zeros(b, ch, ch).type(m.type())
        G = torch.baddbmm(i_, m, m_transposed, beta=0, alpha=1. / (ch * h * w), out=None)
        return G

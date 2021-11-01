import torch

def KD_loss(scores, re_scores, args):
    softmax_scores = torch.log_softmax(scores/args.c1_kdpr, 2)
    softmax_re_tag_teacher = torch.softmax(re_scores/args.c1_kdpr, 2)
    loss_KL = torch.nn.KLDivLoss()(softmax_scores, softmax_re_tag_teacher) * args.c1_kdpr * args.c1_kdpr
    return loss_KL

# this is the 2d case extend to 3d
def PR_loss(scores, re_scores, args):
    log_softmax_scores = torch.log_softmax(scores, 2)
    softmax_scores = torch.softmax(scores, 2)
    product_term = torch.exp(
        re_scores - 1) * args.c1_kdpr  # in PR, c1_prkd stands for the regularization term, higher l2, harder rule constraint
    teacher_score = torch.mul(softmax_scores, product_term)
    softmax_teacher = torch.softmax(teacher_score, 2)
    loss_KL = torch.nn.KLDivLoss()(log_softmax_scores, softmax_teacher)
    return loss_KL

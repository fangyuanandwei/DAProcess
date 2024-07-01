def NRC_option(fea_bank, score_bank,tar_idx, features_test, softmax_out):
    KK = 2
    K  = 3

    with torch.no_grad():
      output_f_norm = F.normalize(features_test)
      output_f_     = output_f_norm.cpu().detach().clone()
      fea_bank[tar_idx]   = output_f_.detach().clone().cpu()
      score_bank[tar_idx] = softmax_out.detach().clone()


def NRC_option(fea_bank, score_bank,tar_idx, features_test, softmax_out):
    KK = 2
    K  = 3
	
	'''
    inputs_target = inputs_test.cuda()
    features_test = netF(inputs_target)
    output        = oldC(features_test)
    softmax_out   = nn.Softmax(dim=1)(output)
	 '''

    with torch.no_grad():
		output_f_norm       = F.normalize(features_test)
		output_f_           = output_f_norm.cpu().detach().clone()
		fea_bank[tar_idx]   = output_f_.detach().clone().cpu()
		score_bank[tar_idx] = softmax_out.detach().clone()
		distance            = output_f_ @ fea_bank.T
		 _, idx_near        = torch.topk(distance, dim=-1,largest=True,k = K + 1)
        idx_near            = idx_near[:, 1:]
        score_near          = score_bank[idx_near]

        fea_near            = fea_bank[idx_near]
        fea_bank_re         = fea_bank.unsqueeze(0).expand(fea_near.shape[0], -1,-1)


        distance_        = torch.bmm(fea_near,fea_bank_re.permute(0, 2,1))
        _, idx_near_near = torch.topk(distance_, dim=-1, largest=True,k=KK + 1)

        idx_near_near = idx_near_near[:, :, 1:]
        tar_idx_      = tar_idx.unsqueeze(-1).unsqueeze(-1)
        match         = (idx_near_near == tar_idx_).sum(-1).float()
        weight        = torch.where( match > 0., match, torch.ones_like(match).fill_(0.1))
        weight_kk     = weight.unsqueeze(-1).expand(-1, -1, KK)
        score_near_kk = score_bank[idx_near_near]
        weight_kk     = weight_kk.contiguous().view(weight_kk.shape[0],-1)
        weight_kk     = weight_kk.fill_(0.1)
        score_near_kk = score_near_kk.contiguous().view(score_near_kk.shape[0], -1, args.class_num)

	output_re = softmax_out.unsqueeze(1).expand(-1, K * KK,-1)
    const     = torch.mean((F.kl_div(output_re, score_near_kk, reduction='none').sum(-1)*weight_kk.cuda()).sum(1))
	loss      = torch.mean(const)

def GSFDA_option(fea_bank, score_bank, tar_idx, features_test, outputs_test):
	
	with torch.no_grad():
		fea_bank[tar_idx].fill_(-0.1)
		output_f_   = F.normalize(features_test).cpu().detach().clone()
		distance    = output_f_ @ fea_bank.t()
		_, idx_near = torch.topk(distance, dim=-1, largest=True, k=2)
		score_near  = score_bank[idx_near]
		score_near  = score_near.permute(0, 2, 1)
	
	softmax_out = nn.Softmax(dim=1)(outputs_test)
    output_re   = softmax_out.unsqueeze(1)
    const       = torch.log(torch.bmm(output_re, score_near)).sum(-1)
    const_loss  = -torch.mean(const)






from .base_trainer import BaseTrainer
import utils
import json
import torch.nn.functional as F
# from basicsr.utils import USMSharp
def is_main_worker(gpu):
    return (gpu <= 0)
import torch

class CombinedTrainer(BaseTrainer):
    """
    CombinedTrainer
    """
    def __init__(self, ddp_gpu, gen, disc, g_optim, d_optim, g_scheduler, d_scheduler,
                 logger, evaluator, cv_loaders, cfg, writer): # cls_char
        super().__init__(ddp_gpu, gen, disc, g_optim, d_optim, g_scheduler, d_scheduler,
                         logger, evaluator, cv_loaders, cfg, writer)


    # region - train
    def train(self, loader, st_step=1, max_step=100000):
        self.step = st_step
        
        self.gen.train()
        if self.disc is not None:
            self.disc.train()

        self.clear_losses()
        
        # --------------------------------------------------------------------------------------------
        with open(self.cfg.structure_tags_path,'r') as  f: stru_map = json.load(f,strict=False)
        with open(self.cfg.cr_mapping_path,'r') as      f: cr_map = json.load(f,strict=False)
        with open(self.cfg.de_path,'r') as f:           de = json.load(f,strict=False)
        # --------------------------------------------------------------------------------------------
        losses = utils.AverageMeters("g_total", "pixel", "disc", "gen","lpips","cross","l1","feat","style_consist","style","fm")
        discs = utils.AverageMeters("real_font", "real_uni", "fake_font", "fake_uni")
        stats = utils.AverageMeters("batch_style", "batch_target")
        
        steps_per_epoch = len(loader)
        num_epoch = int(max_step // len(loader))
        
        # region 학습 시작
        # noise_strength = 0.05
        # while True:
        self.logger.info("Start training ...")
        for epoch in range(num_epoch):
            for _iter, (input_style_ids, input_imgs, input_imgs_ske,
                 trg_style_ids, trg_uni_ids, trg_imgs, 
                 content_imgs, content_imgs_ske, 
                 trg_unis, style_sample_index, trg_sample_index) in enumerate(loader):
                
                # self.epoch = self.step // len(loader)
                batch_size = trg_imgs.shape[0]
                stats.updates({
                    "batch_style": input_imgs.shape[0], # batch*k_shot
                    "batch_target": batch_size
                })
                
                input_style_ids = input_style_ids.cuda()
                input_imgs = input_imgs.cuda()        
                trg_uni_disc_ids = trg_uni_ids.cuda()
                trg_style_ids = trg_style_ids.cuda()
                trg_imgs = trg_imgs.cuda()
                content_imgs = content_imgs.cuda()
                
                # input_styles_unis를 가져오기
                input_styles_unis = [cr_map[i[0]] for i in trg_unis]
                # 타깃 시퀀스에서 각 단위의 구조 정보를 가져와 GPU용 텐서로 변환
                trg_stru_ids      = torch.tensor([stru_map[i[0]] for i in trg_unis]).cuda()
                
                in_stru_ids=[]
                for k in input_styles_unis:
                    for i in range(3):
                        in_stru_ids.append(stru_map[k[i]])
                in_stru_ids = torch.tensor(in_stru_ids).cuda()

                # 타깃 단위와 입력 스타일 단위에서 컴포넌트 정보를 뽑아 리스트로 만드는 과정
                trg_comp_ids = []
                for i in trg_unis:
                    trg_comp_ids.append(de[i[0]])
                    
                in_comp_ids=[]
                for k in input_styles_unis:
                    for i in range(3):
                        in_comp_ids.append(de[k[i]])
                
                if self.cfg.use_half:
                    input_imgs = input_imgs.half()
                    content_imgs = content_imgs.half()

                input_imgs_crose = torch.nn.functional.interpolate(input_imgs,scale_factor=1.2,mode='bilinear')
                input_imgs_crose = input_imgs_crose.cuda()
                input_imgs_fine = torch.nn.functional.interpolate(input_imgs,scale_factor=0.8,mode='bilinear')
                input_imgs_fine = input_imgs_fine.cuda()          
                trg_imgs_crose = torch.nn.functional.interpolate(trg_imgs,scale_factor=1.2,mode='bilinear')
                trg_imgs_fine = torch.nn.functional.interpolate(trg_imgs,scale_factor=0.8,mode='bilinear')
                
                """
                Conditional VQGAN
                
                Real font → Reference style → Content font → Target font 생성
                # Real font: real(진짜) 이미지 trg_imgs가 어떤 font 스타일인지 D가 예측한 결과
                # Reference style: G가 이미지를 생성할 때 참고하는 “스타일 이미지”
                """
                
                ##############################################################
                # infer
                ##############################################################
                # quant, emb_loss, info ,gt_feat= self.gen.vqgan.encode(trg_imgs) #info[2]:[2048]
                # quant, emb_loss, info = self.gen.vqgan.encode(trg_imgs) #info[2]:[2048]
                # tar = self.gen.vqgan.decode(quant)
                # sc_feats = self.gen.encode_write_comb(input_style_ids, style_sample_index, input_imgs, input_imgs_crose,input_imgs_fine,in_stru_ids)
                # out, z_e_x,_,z_q_x ,indice_out= self.gen.read_decode(trg_style_ids, trg_sample_index, content_imgs, trg_stru_ids, in_stru_ids) #fake_img
                # self_infer_imgs, z_e_x_self ,_,z_q_x_self, indice_self= self.gen.infer(trg_style_ids, trg_imgs, trg_imgs_crose, trg_imgs_fine, trg_style_ids, trg_sample_index, trg_sample_index, content_imgs,trg_stru_ids,trg_stru_ids)
                
                """
                * 학습된 VQ-GAN 모델을 불러와 추출
                * quantization = codebook lookup
                * info = codebook index
                """
                quant, _, info = self.gen.vqgan.encode(trg_imgs)
                # # 원본 이미지 재구성
                # _ = self.gen.vqgan.decode(quant)
                
                # handwrite 스타일 메모리에 저장
                sc_feats, comb_style_latent = self.gen.encode_write_comb(input_style_ids, 
                                                                         style_sample_index, 
                                                                         input_imgs, 
                                                                         input_imgs_crose,
                                                                         input_imgs_fine,
                                                                         in_stru_ids)
                # 저장된 handwrite 스타일을 가져와서 content에 적용해 가짜 이미지 생성
                out, _, _, _ , indice_out = self.gen.read_decode(trg_style_ids, 
                                                                 trg_sample_index, 
                                                                 content_imgs, 
                                                                 trg_stru_ids, 
                                                                 in_stru_ids)
                # self-reconstruction
                _, _ ,_, _, indice_self = self.gen.infer(trg_style_ids, 
                                                         trg_imgs, 
                                                         trg_imgs_crose, 
                                                         trg_imgs_fine, 
                                                         trg_style_ids, 
                                                         trg_sample_index, 
                                                         trg_sample_index, 
                                                         content_imgs,
                                                         trg_stru_ids,
                                                         trg_stru_ids)
                
                ################### discriminator ##################
                # get discriminator outputs (skip if discriminator is not present)
                if self.disc is not None:
                    # feature extraction for feature-matching
                    result_real = self.disc(trg_imgs, trg_style_ids, trg_uni_disc_ids, trg_stru_ids)
                    real_font, real_uni, real_stru  = result_real['ret']

                    # [D 학습용]
                    result_fake = self.disc(out.detach(), trg_style_ids, trg_uni_disc_ids, trg_stru_ids)
                    fake_font, fake_uni, fake_stru  = result_fake['ret']

                    self.add_gan_d_loss(real_stru, real_uni, fake_stru, fake_uni)
                    self.d_optim.zero_grad()
                    self.d_backward()
                    self.d_optim.step()
                    self.d_scheduler.step()
                else:
                    real_font = real_uni = real_stru = None
                    fake_font = fake_uni = fake_stru = None

                ################### generator ##################
                # discriminator outputs for generator loss (if available)
                if self.disc is not None:
                    result_fake = self.disc(out, trg_style_ids, trg_uni_disc_ids, trg_stru_ids)
                    fake_font, fake_uni, fake_stru  = result_fake['ret']
                else:
                    fake_font = fake_uni = fake_stru = None
                # fake_feats, fake_rep            = result_fake['feats_out'], result_fake['rep']
                self.add_gan_g_loss(real_font, real_uni, fake_uni, fake_stru)

                # ------------------------------------------------------------------------------
                # self.add_feature_matching_loss(real_feats, fake_feats)
                
                # ✅ 2단계: 붓글씨 스타일 강제
                self.add_style_loss(out, input_imgs)
                # ------------------------------------------------------------------------------

                self.add_l1_loss_only_mainstructure(out, trg_imgs)
                self.add_lpips_loss_only_mainstructure(out, trg_imgs) # 여기선 trg_imgs가 Uhbee 위주이기 때문에 높아야 좋은 것임
                self.add_crossentropy_loss(indice_out, info[2], indice_self)
                
                # ------------ Style consistency loss: 같은 스타일의 multiple scale에서 일관성 보장 ------------
                # style_consistency_loss = F.mse_loss(
                #     F.normalize(sc_feats['last'], p=2, dim=1),
                #     F.normalize(torch.nn.functional.adaptive_avg_pool2d(comb_style_latent, sc_feats['last'].shape[-2:]), p=2, dim=1)
                # ) * 0.5
                style_consistency_loss = F.mse_loss(
                    F.normalize(sc_feats['last'], p=2, dim=1),
                    F.normalize(torch.nn.functional.adaptive_avg_pool2d(comb_style_latent, sc_feats['last'].shape[-2:]), p=2, dim=1)
                ) * 1.5
                self.g_losses['style_consist'] = style_consistency_loss
                # ---------------------------------------------------------------------------------------------
                
                self.g_optim.zero_grad()
                self.g_backward()
                self.g_optim.step()
                self.g_scheduler.step()
                loss_dic = self.clear_losses()
                losses.updates(loss_dic, batch_size)  # accum loss stats

                # EMA g
                # self.accum_g()
                self.accum_g(decay=0.999)
                
                # if self.step % self.cfg['tb_freq'] == 0:
                #     tag_scalar_dic = self.baseplot(losses, discs, stats)

                # if self.step % self.cfg['print_freq'] == 0:
                if self.step % self.cfg['tb_freq'] == 0:
                    self.writer.add_scalars({"optimization/loss/cross entropy": losses.cross.avg}, self.step)
                    self.writer.add_scalars({"optimization/loss/L1": losses.l1.avg}, self.step)
                    self.writer.add_scalars({"optimization/loss/smooth L1": losses.feat.avg}, self.step)
                    self.writer.add_scalars({"optimization/loss/discriminator": losses.disc.avg}, self.step)
                    self.writer.add_scalars({"optimization/loss/generator": losses.gen.avg}, self.step)
                    self.writer.add_scalars({"optimization/loss/style_consist": losses.style_consist.avg}, self.step)
                    self.writer.add_scalars({"optimization/loss/lpips": losses.lpips.avg}, self.step)
                    # self.writer.add_scalars({"optimization/loss/gram style": losses.style.avg}, self.step)
                    # self.writer.add_scalars({"optimization/loss/feature matching": losses.fm.avg}, self.step)
                
                # self.log(losses, discs, stats)
                self.logger.info(
                    f" Epoch: [{epoch:4d}/{num_epoch}]  Step: [{_iter:7d}]/[{steps_per_epoch}]/[{self.step}]/[{max_step}] "
                    f" | cross_entropy: {losses.cross.avg:7.4f},  L1: {losses.l1.avg:7.4f},  Lpips: {losses.lpips.avg:7.4f}, D: {losses.disc.avg:7.3f},  G: {losses.gen.avg:7.3f}"
                    f" | batch_style: {stats.batch_style.avg:5.1f},  batch_target: {stats.batch_target.avg:5.1f}"
                    )
                losses.resets()
                discs.resets()
                stats.resets()

                
                # region validation
                # ------------------------------------------------------------------------------------------------------------------
                if self.step % self.cfg['val_freq'] == 0:
                #     if is_main_worker(self.ddp_gpu):
                    self.logger.info(f"Validation at Epoch = {epoch:.3f}")
                    self.evaluator.cp_validation(self.gen_ema, self.cv_loaders, self.step)
                    
                    rnd_idx = torch.randint(low=0, high=trg_imgs.shape[0], size=(1, ))
                    self.writer.add_image("training/VQGAN input image", trg_imgs[rnd_idx].squeeze(0), self.step)
                    self.writer.add_image("training/styled content image", out[rnd_idx].squeeze(0), self.step)
                    
                    self.writer.add_scalars({"optimization/learning_rate/generator": self.g_optim.param_groups[0]['lr']}, self.step)
                    if self.d_optim is not None:
                        self.writer.add_scalars({"optimization/learning_rate/discriminator": self.d_optim.param_groups[0]['lr']}, self.step)
                            
                    if (self.step >= self.cfg.save_freq) and (self.step % self.cfg['val_freq']==0):
                        self.save(loss_dic['g_total'], self.cfg['save'], self.cfg.get('save_freq', self.cfg['val_freq']))
                # ------------------------------------------------------------------------------------------------------------------

                self.step += 1
                # self.step += 1
                # if self.step >= max_step: break
                # self.step += 1
                
        self.logger.info("Iteration finished.")
        # region 학습 끝


    # def log(self, losses, discs, stats):
    #     # self.logger.info(
    #     #     f" Epoch: [{self.epoch:4d}/{self.num_epoch}]  Step: {self.step:7d} "
    #     #     f" | cross_entropy: {losses.cross.avg:7.4f},  L1: {losses.l1.avg:7.4f},  Lpips: {losses.lpips.avg:7.4f},  Feat: {losses.feat.avg:7.4f},  D: {losses.disc.avg:7.3f},  G: {losses.gen.avg:7.3f}"
    #     #     f" | batch_style: {stats.batch_style.avg:5.1f},  batch_target: {stats.batch_target.avg:5.1f}"
    #     #     )
    #     self.logger.info(
    #         f" Epoch: [{self.epoch:4d}/{self.num_epoch}]  Step: [{self.step:7d}]/[{self.steps_per_epoch}]/[{self.max_step}] "
    #         f" | cross_entropy: {losses.cross.avg:7.4f},  L1: {losses.l1.avg:7.4f},  Lpips: {losses.lpips.avg:7.4f}, D: {losses.disc.avg:7.3f},  G: {losses.gen.avg:7.3f}"
    #         f" | batch_style: {stats.batch_style.avg:5.1f},  batch_target: {stats.batch_target.avg:5.1f}"
    #         )

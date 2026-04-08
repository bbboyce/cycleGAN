"""
Modified CycleGAN Model for Ablation Study
==========================================
Supports flexible loss component enabling/disabling for ablation experiments.

Usage:
    Add to base options:
    --ablation_mode "no_cycle"  # or "no_identity", "cycle_only", etc.
"""

import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class CycleGANAblationModel(BaseModel):
    """
    CycleGAN variant supporting ablation study configurations.
    
    Ablation modes:
    - "full": All loss components (baseline)
    - "no_cycle": Disable cycle consistency loss
    - "no_identity": Disable identity loss
    - "no_gan": Disable GAN loss
    - "cycle_only": Only cycle consistency loss
    - "cycle_and_gan": GAN + Cycle loss, no identity
    - "identity_only": Only identity loss
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument("--lambda_A", type=float, default=10.0, 
                              help="weight for cycle loss (A -> B -> A)")
            parser.add_argument("--lambda_B", type=float, default=10.0, 
                              help="weight for cycle loss (B -> A -> B)")
            parser.add_argument("--lambda_identity", type=float, default=0.5,
                              help="use identity mapping weight")
            parser.add_argument("--ablation_mode", type=str, default="full",
                              choices=["full", "no_cycle", "no_identity", "no_gan", 
                                      "cycle_only", "cycle_and_gan", "identity_only"],
                              help="ablation study mode")

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        
        # Set ablation mode
        self.ablation_mode = getattr(opt, 'ablation_mode', 'full')
        
        # Loss names - will update based on ablation mode
        self.loss_names = ["D_A", "G_A", "cycle_A", "idt_A", "D_B", "G_B", "cycle_B", "idt_B"]
        
        # Visual names
        visual_names_A = ["real_A", "fake_B", "rec_A"]
        visual_names_B = ["real_B", "fake_A", "rec_B"]
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append("idt_B")
            visual_names_B.append("idt_A")

        self.visual_names = visual_names_A + visual_names_B
        
        if self.isTrain:
            self.model_names = ["G_A", "G_B", "D_A", "D_B"]
        else:
            self.model_names = ["G_A", "G_B"]

        # Define networks
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, 
                                       opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG,
                                       opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain)

        if self.isTrain:
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D,
                                           opt.norm, opt.init_type, opt.init_gain)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D,
                                           opt.norm, opt.init_type, opt.init_gain)

        if self.isTrain:
            if opt.lambda_identity > 0.0:
                assert opt.input_nc == opt.output_nc
            
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            
            # Loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            
            # Optimizers
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            
            # Print ablation mode
            print(f"\n{'='*50}")
            print(f"🧪 Ablation Mode: {self.ablation_mode}")
            print(f"{'='*50}")
            self._print_ablation_config()

    def _print_ablation_config(self):
        """Print configuration for current ablation mode"""
        configs = {
            "full": {
                "GAN Loss": "✅ Enabled",
                "Cycle Loss": "✅ Enabled", 
                "Identity Loss": "✅ Enabled"
            },
            "no_cycle": {
                "GAN Loss": "✅ Enabled",
                "Cycle Loss": "❌ Disabled",
                "Identity Loss": "✅ Enabled"
            },
            "no_identity": {
                "GAN Loss": "✅ Enabled",
                "Cycle Loss": "✅ Enabled",
                "Identity Loss": "❌ Disabled"
            },
            "no_gan": {
                "GAN Loss": "❌ Disabled",
                "Cycle Loss": "✅ Enabled",
                "Identity Loss": "✅ Enabled"
            },
            "cycle_only": {
                "GAN Loss": "❌ Disabled",
                "Cycle Loss": "✅ Enabled",
                "Identity Loss": "❌ Disabled"
            },
            "cycle_and_gan": {
                "GAN Loss": "✅ Enabled",
                "Cycle Loss": "✅ Enabled",
                "Identity Loss": "❌ Disabled"
            },
            "identity_only": {
                "GAN Loss": "❌ Disabled",
                "Cycle Loss": "❌ Disabled",
                "Identity Loss": "✅ Enabled"
            }
        }
        
        if self.ablation_mode in configs:
            for component, status in configs[self.ablation_mode].items():
                print(f"{component:20s}: {status}")
        print()

    def set_input(self, input):
        """Unpack input data"""
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self):
        """Run forward pass"""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for discriminator"""
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        if self.ablation_mode == "no_gan":
            self.loss_D_A = torch.tensor(0.0, device=self.device)
            return
            
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        if self.ablation_mode == "no_gan":
            self.loss_D_B = torch.tensor(0.0, device=self.device)
            return
            
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate generator loss"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # Identity loss
        if lambda_idt > 0 and self.ablation_mode != "no_identity":
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss
        if self.ablation_mode == "no_gan":
            self.loss_G_A = 0
            self.loss_G_B = 0
        else:
            self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
            self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # Cycle loss
        if self.ablation_mode == "no_cycle":
            self.loss_cycle_A = 0
            self.loss_cycle_B = 0
        else:
            self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
            self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # Combined loss
        self.loss_G = (
            self.loss_G_A + self.loss_G_B + 
            self.loss_cycle_A + self.loss_cycle_B + 
            self.loss_idt_A + self.loss_idt_B
        )
        self.loss_G.backward()

    def optimize_parameters(self):
        """Optimize network parameters"""
        self.forward()
        
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        
        if self.ablation_mode != "no_gan":
            self.backward_D_A()
            self.backward_D_B()
        else:
            self.loss_D_A = torch.tensor(0.0, device=self.device)
            self.loss_D_B = torch.tensor(0.0, device=self.device)
        
        self.optimizer_D.step()

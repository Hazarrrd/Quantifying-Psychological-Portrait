import torch
import pyro
import pyro.distributions as dist
from .common import N_COUNTIRES, load_joint, empty_df, positive_correlation, QUESTIONS
from .BaseModel import BaseModel
from sklearn.preprocessing import OrdinalEncoder
import torch.distributions.constraints as const
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import seaborn as sns
import pandas as pd
from IPython.display import display
import pomegranate as pg
from torch import nn
import pyro.distributions.constraints as const
from pyro import poutine
from scipy.stats import mode

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()

        self.l1 = nn.Linear(z_dim, hidden_dim)
        self.l2_1 = nn.Linear(hidden_dim, 50 * 5)
        self.l2_2 = nn.Linear(hidden_dim, N_COUNTIRES)
        
        # setup the non-linearities
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, z):
        hidden = self.softplus(self.l1(z))
        anwsers_probs = self.sigmoid(self.l2_1(hidden))
        country_probs = self.softmax(self.l2_2(hidden))
        
        return anwsers_probs, country_probs
    
    def is_trainable(self, trainable):
        for param in self.parameters():
            param.requires_grad = trainable


class FCModel(BaseModel):
    # Parameters of prior trait distributions
    ALPHA_PRIOR = 1.0
    BETA_PRIOR = 1.0
    
    def __init__(self, country_encoder, t=5, h=15, pretrained_decoder_path=None):
        super().__init__(country_encoder)
        self._decoder = Decoder(t, h)
        self._observed_traits = None

        if pretrained_decoder_path is not None:
            self._decoder.load_state_dict(torch.load(pretrained_decoder_path))
            self._decoder.is_trainable(False)
            self._decoder.eval()

        self.t = t
        
        
    def model(self, n_samples=None, anwser_obs=None, obs_anwser_mask=torch.tensor(False), trait_obs_mask=torch.tensor(False), trait_obs=None): 
        if n_samples is None:
            if len(self._observations) > 0:
                n_samples = len(self._observations)
            else:
                n_samples = self._observed_traits.shape[0]
            
        pyro.module("decoder", self._decoder)
        
        with pyro.plate("person", n_samples):
            trait = pyro.sample('trait', dist.Beta(self.ALPHA_PRIOR * torch.ones(self.t), self.BETA_PRIOR * torch.ones(self.t)).to_event(1))

            if trait_obs is not None:
                # Imputation trick. If we used sampling + observation here, we would end up with infinite 
                # gradients (checked that empirically :/)
                # The only parameters learned by the models, are the ones that give trait distribution. If we
                # Condition on traits, there is little value to learn distribution of these conditioned traits.
                # Non-masked traits will learn fine, as they are from non-observer "trait" distribution above 
                trait = trait * (1.0 - trait_obs_mask.long()) + trait_obs * trait_obs_mask.long()            

            question_probs, _ = self._decoder(trait)
            
            question_probs = question_probs.reshape(-1, 50, 5).transpose(1, 0)
            
            with pyro.plate("anwsers", 50):
                anwser = pyro.sample("anwser", dist.Categorical(question_probs))

                if anwser_obs is not None:
                    anwser_observed = pyro.sample('anwser_obs', dist.Categorical(question_probs).mask(obs_anwser_mask), obs=anwser_obs)

                    anwser = anwser.masked_scatter(obs_anwser_mask, anwser_observed)

                return anwser

            
    def traits_guide(self, n_samples=None, anwser_obs=None, obs_anwser_mask=True, trait_obs_mask=torch.tensor(False), trait_obs=None):
        if n_samples is None:
            if len(self._observations) > 0:
                n_samples = len(self._observations)
            else:
                n_samples = self._observed_traits.shape[0]
        
        trait_alphas = pyro.param('trait_alphas', self.ALPHA_PRIOR * torch.ones(n_samples, self.t), constraint = const.positive)
        trait_betas = pyro.param('trait_betas', self.BETA_PRIOR * torch.ones(n_samples, self.t), constraint = const.positive)
        
        with pyro.plate("person", n_samples):
            trait = pyro.sample('trait', dist.Beta(trait_alphas, trait_betas).to_event(1))
    
            question_probs, _ = self._decoder(trait)
            
            question_probs = question_probs.reshape(-1, 50, 5).transpose(1, 0)
            
            with pyro.plate("anwsers", 50):
                anwser = pyro.sample("anwser", dist.Categorical(question_probs))
                
                
    def infer(self, num_steps = 1_000, train=False, verbose=False):  
        pyro.clear_param_store()

        conditioned_model, conditioned_guide = self.model_conditioned(anwers=self._observations, traits=self._observed_traits)

        self._decoder.is_trainable(train)
        
        pyro.enable_validation(True)
        svi = pyro.infer.SVI(model=conditioned_model,
                             guide=conditioned_guide,
                             optim=pyro.optim.Adam({"lr": 1e-2}),
                             loss=pyro.infer.TraceGraph_ELBO())

        losses = []
        alphas = []
        for t in tqdm(range(num_steps)):
            losses.append(svi.step())
            alphas.append(pyro.param('trait_alphas').detach().numpy()[0, 0])

        if verbose:
            plt.plot(losses)
            plt.title("ELBO")
            plt.xlabel("step")
            plt.ylabel("loss")
            plt.show()
            
            plt.plot(alphas)
            plt.title("Alphas")
            plt.xlabel("step")
            plt.ylabel("Alpha")
            plt.show()

    def reset_observations(self):
        super().reset_observations()
        self._observed_traits = None
    
    def model_conditioned(self, n_samples=None, anwers=None, traits=None):
        anwer_obs = None
        trait_obs = None
        
        if anwers is not None and len(anwers) > 0:
            anwer_obs = anwers.copy(deep=True)
            anwer_obs = anwer_obs.fillna(0.0)
            anwer_obs = torch.tensor(anwer_obs.drop('country', axis=1).values.reshape(-1, 50).transpose([1, 0])).long()

        if traits is not None:
            trait_obs = traits.copy()
            trait_obs = np.nan_to_num(trait_obs)
            trait_obs = torch.tensor(trait_obs).float()

        def model_masked(n_samples = n_samples):
            return self.model(
                n_samples = n_samples,
                anwser_obs = anwer_obs,
                obs_anwser_mask = torch.tensor(self.anwser_mask().reshape(-1, 50).transpose([1, 0])),
                trait_obs_mask = torch.tensor(self.traits_mask()),
                trait_obs = trait_obs
            )

        def guide_masked(n_samples = n_samples):
            return self.traits_guide(
                n_samples = n_samples,
                anwser_obs = anwer_obs,
                obs_anwser_mask = torch.tensor(self.anwser_mask().reshape(-1, 50).transpose([1, 0])),
                trait_obs_mask = torch.tensor(self.traits_mask()),
                trait_obs = trait_obs
            )

        return model_masked, guide_masked

    
    def anwsers_given_traits(self, traits: np.ndarray):
        # No need to do inference, as we condition on input variables
        trait = torch.tensor(traits.reshape(1, self.t), dtype=torch.float32).expand(1_000, -1)

        conditional_model = pyro.condition(self.model, data={
            'trait': trait
        })

        return conditional_model(n_samples = 1_000)

        
    def sample(self, n_samples):
        conditioned_model, conditioned_guide = self.model_conditioned(anwers=self._observations, traits=self._observed_traits)

        return pyro.infer.Predictive(conditioned_model, guide=conditioned_guide, num_samples=n_samples)()

    def observe_traits(self, traits):
        self._observed_traits = np.copy(traits)

    def traits_mask(self):
        if self._observed_traits is None:
            return False

        return pd.isnull(self._observed_traits) == False

    def predict_anwser(self, data, steps=2_000, n_samples=2_000):
        self.reset_observations()
        self.observe(data=data)
        self.infer(steps)

        samples = self.sample(n_samples)['anwser'].detach().numpy() + 1
        reconstructed = mode(samples).mode.reshape(50, -1).transpose(1, 0)

        result = data.copy()
        result.iloc[:, :50] = reconstructed

        return result

    def log_prob(self, data):
        self.reset_observations()
        self.observe(data=data)
        self.infer(3_000, train=False)

        conditioned_model, conditioned_guide = self.model_conditioned(anwers=self._observations, traits=self._observed_traits)

        guide_trace = poutine.trace(conditioned_guide).get_trace()
        model_trace = poutine.trace(
            poutine.replay(conditioned_model, trace=guide_trace)
        )

        return (model_trace.get_trace().log_prob_sum(lambda x,y: x is 'anwser')).detach().numpy()
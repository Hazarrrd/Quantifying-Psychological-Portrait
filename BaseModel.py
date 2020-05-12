from common import N_COUNTIRES, load_joint, empty_df, positive_correlation, QUESTIONS
from sklearn.preprocessing import OrdinalEncoder
import torch.distributions.constraints as const
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import seaborn as sns
import pandas as pd
from IPython.display import display

class BaseModel:
    def __init__(self, country_encoder=None):           
        self._observations = self._empty_observations()
        self._encoder = country_encoder
        
    def _empty_observations(self):
        initial_obs = {
            **{
                key: []
                for key in QUESTIONS.keys()
            },
            'country': []
        }

        return pd.DataFrame(initial_obs)
        
        
    def reset_observations(self):
        self._observations = self._empty_observations()
        
        
    def full_mask(self):
        return self._observations.values != None
    
    
    def country_mask(self):
        return self.full_mask()[:, -1]
    
    
    def anwser_mask(self):
        return self.full_mask()[:, :-1]
    
        
    def _add_user(self, user: str):
        initial_obs = {
            **{
                key: None
                for key in QUESTIONS.keys()
            },
            'country': None
        }
        
        self._observations = pd.concat([
            self._observations, 
            pd.DataFrame(initial_obs, index=[user])
        ])
        
                         
    def observe(self, user=None, x=None, data=None) -> None:        
        if data is not None:
            observations = data.copy(deep=True)
            observations.iloc[:, :-1] = observations.iloc[:, :-1].apply(lambda x: np.clip(x - 1.0, 0, 4) if x is not None else x)
            if self._encoder is not None:
                observations['country'] = observations['country'].apply(lambda x: self._encoder.transform([[x]])[0,0] if (x is not None) else None)
                
            self._observations = pd.concat([
                self._observations,
                observations
            ])
        else:
            if user not in self._observations.index:
                self._add_user(user)

            for key, value in x.items():
                if key != 'country':
                    # We map values 1 - 5 into 0 - 4
                    self._observations.loc[user, key] = np.clip(value - 1.0, 0, 4)
                else:
                    if self._encoder is not None:
                        value = self._encoder.transform([[value]])[0,0]
                    self._observations.loc[user, key] = value
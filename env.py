from abc import abstractmethod
import numpy as np
import torch 

class Env: 
    def __init__(self, observation_shape: np.ndarray, action_shape: np.ndarray):
        self.observation_shape = observation_shape
        self.action_shape = action_shape
    
    @property
    def is_multi_discrete(self):
        return self.action_shape.size > 1
    
    @abstractmethod
    def _run_step(self, action) -> tuple[np.ndarray, float, bool, object]:
        raise NotImplementedError
    
    @abstractmethod
    def _run_reset(self, action) -> np.ndarray:
        raise NotImplementedError

    def step(self, action):
        if torch.is_tensor(action):
            action = torch.flatten(action).cpu().numpy()
        if type(action) is not int and action.size == 1: 
            action = action.item()
        observation, reward, done, info = self._run_step(action)
        observation = torch.from_numpy(observation).unsqueeze(dim=0).float()
        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        observation = self._run_reset()
        return torch.from_numpy(observation).unsqueeze(dim=0).float()
    
    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


    @abstractmethod
    def randomise_seed(self) -> None:
        raise NotImplementedError
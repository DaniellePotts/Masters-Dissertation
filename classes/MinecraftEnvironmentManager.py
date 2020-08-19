import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
class MinecraftEnvironmentManager():
    def __init__(self, client, device, environment):
        self.device = device
        self.client = client
        self.environment = environment
       
        join_tokens = self.start_environment(environment, 10000, ["MarLo-Agent-0"])
        
        assert len(join_tokens) == 1
        join_token = join_tokens[0]

        self.env = client.init(join_token)
        
       
        # self.env.reset()
        # self.current_screen = None
        self.done = False
    
    def launch_enviroment(self):
        import malmo.minecraftbootstrap; malmo.minecraftbootstrap.launch_minecraft()
    def start_environment(self, environment, port, agent_names):
        client_pool = [('127.0.0.1', port)]
        params = {"client_pool": client_pool,"agent_names" :agent_names}
        join_tokens = self.client.make(environment,
                      params=params)
        return join_tokens
    def basic_run(self):
        observation = self.env.reset()

        self.done = False
        while not self.done:
          _action = self.env.action_space.sample()
          obs, reward, self.done, info = self.env.step(_action)
          print("reward:", reward)
          print("done:", self.done)
          print("info", info)
        self.env.close()
    def reset(self):
        self.env.reset()
        self.current_screen = None
        
    def close(self):
        self.env.close()
        
    def render(self, mode='human'):
        return self.env.render(mode)
        
    def num_actions_available(self):
        return self.env.action_space.n
        
    def take_action(self, action):        
        _, reward, self.done, _ = self.env.step(action.item())
        return torch.tensor([reward], device=self.device)
    
    def just_starting(self):
        return self.current_screen is None
    
    def get_state(self):
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_screen = s2
            return s2 - s1
    
    def get_screen_height(self):
        screen = self.get_processed_screen()
        return screen.shape[2]
    
    def get_screen_width(self):
        screen = self.get_processed_screen()
        return screen.shape[3]
       
    def get_processed_screen(self):
        screen = self.render('rgb_array').transpose((2, 0, 1)) # PyTorch expects CHW
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)
    
    def crop_screen(self, screen):
        screen_height = screen.shape[1]
        
        # Strip off top and bottom
        top = int(screen_height * 0.4)
        bottom = int(screen_height * 0.8)
        screen = screen[:, top:bottom, :]
        return screen
    
    def transform_screen_data(self, screen):       
        # Convert to float, rescale, convert to tensor
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        
        # Use torchvision package to compose image transforms
        resize = T.Compose([
            T.ToPILImage()
            ,T.Resize((40,90))
            ,T.ToTensor()
        ])
        
        return resize(screen).unsqueeze(0).to(self.device) # add a batch dimension (BCHW)
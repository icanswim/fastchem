from datetime import datetime
import logging
import random
import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'

from torch import device, nn, cuda, optim, no_grad, save, load, cat
from torch.utils.data import Sampler, DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Learn():
    
    def __init__(self, Dataset, Model, Sampler, Optimizer=None, Criterion=None, 
                 model_params={}, ds_params={}, opt_params={}, crit_params={}, 
                 batch_size=1, epochs=1, save_model=False, load_model=False, 
                 adapt=False):
        """
        save_model = True/False
        load_model = False/'./models/savedmodel.pth'
        Criterion = None implies inference mode.
        adapt = False/(dataset input shape, model input shape) 
        """
        logging.basicConfig(filename='./logs/quantum.log', level=20)
        start = datetime.now()
        logging.info('New experiment...\n\n model: {}, start time: {}'.format(
                                        Model, start.strftime('%Y%m%d_%H%M')))
        self.bs = batch_size
        self.ds = Dataset(**ds_params)
        logging.info('dataset: {}\n{}'.format(type(self.ds), ds_params))
        logging.info('epochs: {}, batch_size: {}, save_model: {}, load_model: {}'.format(
                                    epochs, batch_size, save_model, load_model))
        print('{} dataset created...'.format(type(self.ds)))
        
        if load_model: 
            try:
                model = Model(embeddings=self.ds.embeddings, **model_params)
                model.load_state_dict(load(load_model))
                print('model loaded from state_dict...')
            except:
                model = load(load_model)
                print('model loaded from pickle...')
        else: 
            model = Model(embeddings=self.ds.embeddings, **model_params)
        if adapt: model.adapt(adapt)
        self.model = model.to('cuda:0')
        logging.info(self.model.children)
        
        self.sampler = Sampler(self.ds.ds_idx)
        
        if Criterion:
            self.criterion = Criterion(**crit_params).to('cuda:0')
            self.opt = Optimizer(self.model.parameters(), **opt_params)

            self.train_log, self.val_log = [], []
            for e in range(epochs):
                self.sampler.sample_train_val_idx()
                self.run('train')
                with no_grad():
                    self.run('val')
                if e % 1 == 0:  
                    print('epoch: {} of {}, train loss: {}, val loss: {}'.format(
                                e, epochs, self.train_log[-1], self.val_log[-1]))     
            with no_grad():
                self.run('test')
                
            pd.DataFrame(zip(self.train_log, self.val_log)).to_csv(
                                        './logs/'+start.strftime("%Y%m%d_%H%M"))
            self.view_log('./logs/'+start.strftime('%Y%m%d_%H%M'))
        else: 
            with no_grad():
                self.run('infer')
        
        elapsed = datetime.now() - start
        if save_model and not adapt: save(self.model.state_dict(), './models/{}.pth'.format(
                                                            start.strftime("%Y%m%d_%H%M")))
        if save_model and adapt: save(self.model, './models/{}.pth'.format(
                                                            start.strftime("%Y%m%d_%H%M")))
        logging.info('learning time: {} \n'.format(elapsed))
        print('learning time: {}'.format(elapsed))
        
    def run(self, flag): 
        
        e_loss, i, predictions = 0, 0, []
        
        self.model.training = True
        if flag != 'train': self.model.training = False
        
        dataloader = DataLoader(self.ds, batch_size=self.bs, shuffle=False, 
                                sampler=self.sampler(flag=flag), batch_sampler=None, 
                                num_workers=8, collate_fn=None, pin_memory=True, 
                                drop_last=False, timeout=0, worker_init_fn=None)
  
        def to_cuda(data):
            if len(data) == 0: return []
            else: return data.to('cuda:0', non_blocking=True)
     
        for x_con, x_cat, y in dataloader:
            i += self.bs
            x_con = to_cuda(x_con)
            x_cat = to_cuda(x_cat)
            y_pred = self.model(x_con, x_cat)
            
            if flag == 'infer':
                y = np.reshape(y, (-1, 1)) # y = 'id'
                y_pred = np.reshape(y_pred.data.to('cpu').numpy(), (-1, 1))
                predictions.append(np.concatenate((y, y_pred), axis = 1)) 
            else:
                y = to_cuda(y)           
                b_loss = self.criterion(y_pred, y)
                e_loss += b_loss.item()
                if flag == 'train':
                    self.opt.zero_grad()
                    b_loss.backward()
                    self.opt.step()

        if flag == 'train': self.train_log.append(e_loss/i)
        if flag == 'val': self.val_log.append(e_loss/i)
        if flag == 'test':  
            logging.info('test loss: {}'.format(e_loss/i))
            print('test loss: {}'.format(e_loss/i))
        if flag == 'infer': 
            logging.info('inference complete')
            predictions = np.concatenate(predictions, axis=0)
            predictions = np.reshape(predictions, (-1, 2))
            self.predictions = pd.DataFrame(predictions, columns=['id','scalar_coupling_constant'])
            self.predictions['id'] = self.predictions['id'].astype('int64')
            self.predictions.to_csv('quantum_inference.csv', header=True, index=False)
            print('inference complete and saved to csv...')

    @classmethod    
    def view_log(cls, log_file):
        log = pd.read_csv(log_file)
        log.iloc[:,1:3].plot(logy=True)
        plt.show()

                                         
class Selector(Sampler):
   
    def __init__(self, dataset_idx, splits=(.1,.8)):
        self.dataset_idx = dataset_idx
        self.splits = splits # (n test/n total, n train/n total), remainder = validation ratio (bootstrapped)
        self.test_idx = random.sample(self.dataset_idx, int(len(self.dataset_idx)*splits[0]))
        self.sample_train_val_idx()

    def __iter__(self):
        if self.flag == 'train':
            return iter(self.train_idx)
        if self.flag == 'val':
            return iter(self.val_idx)
        if self.flag == 'test':
            return iter(self.test_idx)
        if self.flag == 'infer':
            return iter(self.dataset_idx)

    def __len__(self):
        if self.flag == 'train':
            return len(self.train_idx)
        if self.flag == 'val':
            return len(self.val_idx)
        if self.flag == 'test':
            return len(self.test_idx) 
        if self.flag == 'infer':
            return len(self.dataset_idx)
        
    def __call__(self, flag):
        self.flag = flag
        return self
    
    def sample_train_val_idx(self):
        train_val_idx = [i for i in self.dataset_idx if i != self.test_idx]
        self.train_idx = random.sample(train_val_idx, int(len(self.dataset_idx)*self.splits[1]))
        self.val_idx = [i for i in train_val_idx if i != self.train_idx]
        
        

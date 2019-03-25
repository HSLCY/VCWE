import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse

from data_reader import DataReader, Word2vecDataset
from model import VCWEModel
from optimization import VCWEAdam

class Word2VecTrainer:
    def __init__(self, input_file, vocabulary_file, img_data_file, char2ix_file, output_dir, maxwordlength, emb_dimension, line_batch_size, sample_batch_size, 
                neg_num, window_size, discard, epochs, initial_lr, seed):
                 
        torch.manual_seed(seed)
        self.img_data = np.load(img_data_file)
        self.data = DataReader(input_file, vocabulary_file, char2ix_file, maxwordlength, discard, seed)
        dataset = Word2vecDataset(self.data, window_size, sample_batch_size, neg_num)
        self.dataloader = DataLoader(dataset, batch_size=line_batch_size,
                                     shuffle=True, num_workers=0, collate_fn=dataset.collate)

        self.output_dir = output_dir
        self.emb_size = len(self.data.word2id)
        self.char_size = len(self.data.char2id)+1       #5031
        self.emb_dimension = emb_dimension
        self.line_batch_size = line_batch_size
        self.epochs = epochs
        self.initial_lr = initial_lr
        self.VCWE_model = VCWEModel(self.emb_size, self.emb_dimension, self.data.wordid2charid, self.char_size)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.num_train_steps= int(len(self.dataloader) * self.epochs)
        if self.use_cuda:
            self.VCWE_model.cuda()


    def train(self):
        self.img_data = torch.from_numpy(self.img_data).to(self.device)
        
        no_decay = ['bias']
        optimizer_parameters = [
             {'params': [p for n, p in self.VCWE_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
             {'params': [p for n, p in self.VCWE_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
             ]
		
        print("num_train_steps=",self.num_train_steps)
        optimizer = VCWEAdam(optimizer_parameters,
                             lr=self.initial_lr,
                             warmup=0.1,
                             t_total=self.num_train_steps)
                             
        for epoch in range(self.epochs):

            print("Epoch: " + str(epoch + 1))
       

            running_loss = 0.0
            for i, sample_batched in enumerate(tqdm(self.dataloader)):

                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)
                    lengths = sample_batched[3].to(self.device)

                    optimizer.zero_grad()
                    loss = self.VCWE_model.forward(pos_u, pos_v, neg_v, self.img_data)
                    running_loss += loss.item()

                    loss.backward()
                    optimizer.step()

                    if i > 0 and i % 1000 == 0:
                        print('loss=', running_loss/1000)
                        running_loss=0.0


            if (epoch+1) % 5 == 0 or (epoch+1) == self.epochs:
                self.VCWE_model.save_embedding(self.data.id2word, self.output_dir+"zh_wiki_VCWE_ep"+str(epoch+1)+".txt")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file",
                        default="./data/zh_wiki.txt",
                        type=str,
                        required=True,
                        help="The input file that the VCWE model was trained on.")    
    parser.add_argument("--vocab_file",
                        default="./data/vocabulary.txt",
                        type=str,
                        required=True,
                        help="The vocabulary file that the VCWE model was trained on.")
    parser.add_argument("--img_data_file",
                        default="./data/char_img_sub_mean.npy",
                        type=str,
                        help="The image data file that the VCWE model was trained on.") 
    parser.add_argument("--char2ix_file",
                        default="./data/char2ix.npz",
                        type=str,
                        help="The character-to-index file corespond to the image data file.")                         
    parser.add_argument("--output_dir",
                        default="./embedding/",
                        type=str,
                        help="The output directory where the embedding file will be written.")   
    parser.add_argument("--line_batch_size",
                        default=32,
                        type=int,
                        help="Batch size for lines.")
    parser.add_argument("--sample_batch_size",
                        default=128,
                        type=int,
                        help="Batch size for samples in a line.")      
    parser.add_argument("--emb_dim",
                        default=100,
                        type=int,
                        help="Embedding dimensions.")     
    parser.add_argument("--maxwordlength",
                        default=5,
                        type=int,
                        help="The maximum number of characters in a word.")  
    parser.add_argument("--neg_num",
                        default=5,
                        type=int,
                        help="The number of negative samplings.") 
    parser.add_argument("--window_size",
                        default=5,
                        type=int,
                        help="The window size.") 
    parser.add_argument("--discard",
                        default=1e-5,
                        type=int,
                        help="The sub-sampling threshold.") 
    parser.add_argument("--learning_rate",
                        default=0.001,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=50,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', 
                        type=int, 
                        default=12345,
                        help="random seed for initialization")                        
    args = parser.parse_args()
    w2v = Word2VecTrainer(input_file = args.input_file, \
                          vocabulary_file = args.vocab_file, \
                          img_data_file = args.img_data_file, \
                          char2ix_file = args.char2ix_file, \
                          output_dir = args.output_dir, 
                          maxwordlength = args.maxwordlength,
                          emb_dimension = args.emb_dim, 
                          line_batch_size = args.line_batch_size,
                          sample_batch_size = args.sample_batch_size,
                          neg_num = args.neg_num,
                          window_size = args.window_size,
                          discard = args.discard,
                          epochs = args.num_train_epochs,
                          initial_lr = args.learning_rate,
                          seed = args.seed)
    w2v.train()
    
if __name__ == "__main__":
    main()
                

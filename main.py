import shutil
import pickle
import torch
from model import network
from data_loader_spec import wave_spec
import argparse
from torchvision import transforms
import os 
import torch.nn as nn 
from torch.autograd import Variable
import torch.optim as optim

def save_checkpoint(state, is_best, filename,savetrainloss,savetraincorrects,savevalloss,savevalcorrects):
    torch.save(state, filename)

    savetrainloss_name = open('trainloss' + str(epoch) + '.pkl','wb')
    savetraincorrects_name = open('traincorrects' + str(epoch) + '.pkl','wb')
    savevalloss_name = open('valloss' + str(epoch) + '.pkl','wb')
    savevalcorrects_name = open('valcorrects' + str(epoch) + '.pkl','wb')

    pickle.dump(savetrainloss,savetrainloss_name)
    pickle.dump(savetraincorrects,savetraincorrects_name)
    pickle.dump(savevalloss,savevalloss_name)
    pickle.dump(savevalcorrects,savevalcorrects_name)

    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
        print("New Best Model Found")

def model_run(phase,model,inputs,labels,criterion,optimizer):
        
    if phase == 'train':
        model.train(True)
    else:
        model.eval()

    if phase == 'train':

    	if torch.cuda.is_available():
    		inputs, labels = Variable(inputs.cuda(), requires_grad = False), Variable(labels.cuda(), requires_grad = False)
    	else:
        	inputs, labels = Variable(inputs, requires_grad = False), Variable(labels, requires_grad = False)
    else:

    	if torch.cuda.is_available():
    		inputs, labels = Variable(inputs.cuda(), volatile = True), Variable(labels.cuda(), volatile= True)
    	else:
        	inputs, labels = Variable(inputs, volatile = True), Variable(labels, volatile= True)


    optimizer.zero_grad()
    outputs = model(inputs)
    if phase == 'val':
        import pdb
        pdb.set_trace()
    labels = labels.type(torch.cuda.FloatTensor)
    outputs = torch.squeeze(outputs)

    loss = criterion(outputs, labels)
    
    pred = outputs.data.clone()
    pred[pred>0.5] = 1
    pred[pred<=0.5] = 0

    if phase=='train':
        loss.backward()
        optimizer.step()

    corrects = torch.sum(pred==labels.data)
    return loss.cpu().data[0],corrects,outputs


parser = argparse.ArgumentParser(description='PyTorch Speech Recognition')

parser.add_argument('--lr','--learning_rate',type=float,default=0.001,help='initial learning rate')
parser.add_argument('--lr_de','--lr_decay',type=int,default=30,help='learning rate decay epoch')
parser.add_argument('--checkpoint',type=str,default='')
parser.add_argument('--wd','--weightdecay',type=float,default=0)
args = parser.parse_args()

print("learning_rate: {0}, decay:{1}, checkpoint:{2}".format(args.lr,args.lr_de,args.checkpoint))

data_transforms = {
    'train': transforms.Compose([
    	transforms.Scale((129,71)),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ]),
    'val': transforms.Compose([
    	transforms.Scale((129,71)),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])
}

print ("....Initializing data sampler.....")
data_dir = os.path.expanduser('data')
dsets = {x: wave_spec(os.path.join(data_dir, x), trans=data_transforms[x])
         for x in ['train', 'val']}

dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=25, num_workers=10,shuffle=True) 
                for x in ['train', 'val']}

print ("....Loading Model.....")
model_ft = network()

if torch.cuda.is_available():
	model_ft = model_ft.cuda()

print ("....Model loaded....")

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

if args.checkpoint:
    state = torch.load(args.checkpoint)
    shutil.copyfile(args.checkpoint,'prev_' + args.checkpoint)
    model_ft.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    start_epoch = state['epoch']
    best_loss = state['best_loss']

    print("checkpoint Loaded star_epoch = {0},best_loss= {1}".format(start_epoch,best_loss))
    del state
    
    savetrainloss_name = open('trainloss' + str(start_epoch) + '.pkl','rb')
    savetraincorrects_name = open('traincorrects' + str(start_epoch) + '.pkl','rb')
    savevalloss_name = open('valloss' + str(start_epoch) + '.pkl','rb')
    savevalcorrects_name = open('valcorrects' + str(start_epoch) + '.pkl','rb')

    savetrainloss = pickle.load(savetrainloss_name)
    savetraincorrects = pickle.load(savetraincorrects_name)
    savevalloss = pickle.load(savevalloss_name)
    savevalcorrects = pickle.load(savevalcorrects_name)

    start_epoch+=1

else:
    start_epoch = 0
    best_loss = float('inf')
    savetrainloss = {}
    savetraincorrects = {}
    savevalloss = {}
    savevalcorrects = {}

for epoch in range(start_epoch,500):
        
    trainloss = 0.0
    traincorrects = 0.0

    for i,data in enumerate(dset_loaders['train'],1):
        input_speech,label = data['image'],data['label']
        loss,correct,_ = model_run('train',model_ft, input_speech, label, criterion, optimizer)
        trainloss += loss
        traincorrects += correct

    trainloss = trainloss/i
    traincorrects = traincorrects/i

    savetrainloss[epoch] = trainloss
    savetraincorrects[epoch] = traincorrects

    valloss = 0.0
    valcorrects = 0.0

    for i,data in enumerate(dset_loaders['val'],1):
        input_speech,label = data['image'],data['label'] 
        loss,correct,_ = model_run('val',model_ft, input_speech, label, criterion, optimizer)
        valloss += loss
        valcorrects += correct

    valloss = valloss/i
    valcorrects = valcorrects/i
   
    savevalloss[epoch] = valloss
    savevalcorrects[epoch] = valcorrects
 
    if valloss < best_loss:
        best_loss = valloss
        is_best = 1
    else:
        is_best = 0

    save_checkpoint({'epoch': epoch,
    'model': model_ft.state_dict(),
    'optimizer': optimizer.state_dict(),
    'best_loss': best_loss},is_best,'checkpoint_ep%d.pth.tar'%(epoch),savetrainloss,savetraincorrects,savevalloss,savevalcorrects)
    
    print ('Epoch = {0}, TrainingLoss = {1}, Train_corrects = {2},val Loss = {3}, val_corrects{4}'.format(epoch,trainloss,traincorrects,valloss,valcorrects))

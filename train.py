from dataset_explore import load_data
from model import  Net
from torchvision import transforms
import torch
import wandb
torch.manual_seed(17)
from tqdm import tqdm
import torchmetrics
from einops import reduce
from torch.autograd import Variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sens_spec(c_matrix_train):
    Sensitivity_train = 100*c_matrix_train[0][0].item() / (c_matrix_train[0][1].item() + c_matrix_train[0][0].item())
    Specificity_train = 100*c_matrix_train[1][1].item()/(c_matrix_train[1][1].item() + c_matrix_train[1][0].item())
    return Sensitivity_train, Specificity_train


def train():
    batch_size = 64
    training= True
    no_classes= 4
    net =  Net(no_classes)
    net.to(device)
    param  = net.parameters()
    dd_augmentation = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])
    dd_augmentation_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])
    data_load_train =  load_data(batch_size, "train", dd_augmentation)
    data_load_val  = load_data(batch_size, "validation",dd_augmentation_val)
    criteria = torch.nn.CrossEntropyLoss()
    optimizer  = torch.optim.Adam(net.parameters(), lr=0.001)
    #optimizer_ft  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    episode = 10
    #Confusion_train_matrix  = torchmetrics.ConfusionMatrix(num_classes=2)
    #Confusion_val_matrix    = torchmetrics.ConfusionMatrix(num_classes=2)
    # training
    precision_train = torchmetrics.Precision().to(device)
    recall_train  = torchmetrics.Recall().to(device)
    # validation
    precision_val = torchmetrics.Precision().to(device)
    recall_val    = torchmetrics.Recall().to(device)
    wandb.init(project="covid_detection")
    for epoch in tqdm(range(episode)):
        running_loss = 0.0
        for i,data in enumerate(data_load_train):
            x, y  = data
            y = y.squeeze().type(torch.LongTensor)
            x =Variable(x.to(device))
            y =Variable(y.to(device))
            optimizer.zero_grad()
            y_predicted  = net(x)
            loss = criteria(y_predicted,y)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(y_predicted, 1)
            # traingin Metrics
            precision_train(predicted.type(torch.int8), y.type(torch.int8))
            recall_train(predicted.type(torch.int8), y.type(torch.int8))
            running_loss += loss.item()
        if epoch %1 == 0 and epoch!=0:
            net.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for data_val in data_load_val:
                    x_val, y_val  = data_val
                    y_val = y_val.squeeze().type(torch.LongTensor)
                    x_val = Variable(x_val.to(device))
                    y_val = Variable(y_val.to(device))
                    y_predicted = net(x_val)
                    loss_val = criteria(y_predicted, y_val)
                    _,predicted_val  =  torch.max(y_predicted, 1)
                    # now we can calculate the metrics
                    precision_val(predicted_val.type(torch.int8), y_val.type(torch.int8))
                    recall_val(predicted_val.type(torch.int8), y_val.type(torch.int8))
                    running_val_loss += loss_val.item()
            # for training
            #sensitivity_train, specificity_train = sens_spec(Confusion_train_matrix.compute())
            # for validation
            #sensitivity_val,   specificity_val   = sens_spec(Confusion_val_matrix.compute())
            #print("Training Loss: %.2f Training { Sensitivity: %.2f and Specificity: %.2f}, Val Loss: %.2f {Val Sensitivity: %.2f and Specificity: %.2f" %(running_loss/10, sensitivity_train, specificity_train, running_val_loss/val_count, sensitivity_val, specificity_val))
            wandb.log({"train_loss":running_loss/len(data_load_train), "train_precision": precision_train.compute(), "train_recall":recall_train.compute(), "val_loss":running_val_loss/len(data_load_val), "val_preision":precision_val.compute(), "val_recall":recall_val.compute()})
            print("Training {Loss: {%.2f}, Precision & Recall {%.2f , %.2f} }, Validation { Loss:{%.2f}, Precision & Recall {%.2f , %.2f } }  "%(running_loss/len(data_load_train), precision_train.compute(), recall_train.compute(),running_val_loss/len(data_load_val),precision_val.compute(), recall_val.compute()))
           
        precision_train.reset()
        recall_train.reset()
        recall_val.reset()
        precision_val.reset()

        if epoch % 5 ==0 and epoch !=0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
            },"model_weights/covid_detection.pth")
        # resetting the metrics
        

        #optimizer_ft.step()
if __name__ == "__main__":
    train()

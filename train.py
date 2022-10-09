from config import *
import torch.utils.data as dataloader
from dataloader import H5Dataset
import torch.optim as optim
from loss_func import combined_loss, dice_loss, dice
import time
from datetime import datetime

# --------------------------CUDA check-----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------------init Seg---------------
#numinitfeatures=32
model = DenseResNet(num_init_features=32, growth_rate=16, block_config=(4, 4, 4, 4), drop_rate=0.2, num_classes=num_classes).to(device)
# --------------Loss---------------------------

loss_criteria = combined_loss({'batch_dice': False, 'smooth': 1e-5, 'do_bg': False, 'square': False}, {}).cuda()
# loss_criteria = dice_loss({'batch_dice': False, 'smooth': 1e-5, 'do_bg': False, 'square': False}, {}).cuda()
# loss_criteria = nn.CrossEntropyLoss().cuda()

# setup optimizer
### invertir el orden de acuerdo al error mostrado en la consola
optimizer = optim.Adam(model.parameters(), lr=lr_init, weight_decay=6e-4, betas=(0.97, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_stepsize, gamma=0.1)
# --------------Start Training and Validation ---------------------------
if __name__ == '__main__':
    checkpt = './checkpoints'
    if not os.path.exists(checkpt):
        os.makedirs(checkpt)
    #-----------------------Training--------------------------------------
    mri_data_train = H5Dataset("D:/EduardoCavieres/TesisMagister/Udense_RN_128/data_train_128", mode='train')
    trainloader = dataloader.DataLoader(mri_data_train, batch_size=batch_train, shuffle=True)
    mri_data_val = H5Dataset("D:/EduardoCavieres/TesisMagister/Udense_RN_128/data_val_128", mode='val')
    valloader = dataloader.DataLoader(mri_data_val, batch_size=1, shuffle=False)
    
    hora = datetime.now()
    hora = hora.strftime(("%d-%m-%Y_%H-%M-%S"))	
    
    dir_train = 'D:/EduardoCavieres/TesisMagister/Resultados/Entrenamiento_%s' %hora
    if not os.path.exists(dir_train):
        os.makedirs(dir_train)
    # loss_arr = []

    
    f1 = open('%s/_loss_%s.txt' %(dir_train,hora), 'a+')
    f2 = open('%s/_out_%s.txt' %(dir_train,hora) , 'a+')
    print('              Clock |        LR |   Epoch |           Loss |        Val DSC |')
    f2.write( 'Fecha inicio: %s \n' %hora )
    f2.write( 'Train Config: \n' )
    f2.write( 'num_epoch = %s ; lr_init = %s ; lr_stepsize = %s ; \n batch_train = %s ; num_classes= %s ; crop_size = %s \n \n' %(num_epoch, lr_init, lr_stepsize, batch_train, num_classes, crop_size))
    f2.write( '              Clock |        LR |   Epoch |           Loss |        Val DSC |        net DSC |        pe DSC |        et DSC |\n' )

    for epoch in range (num_epoch + 1):
        scheduler.step(epoch)
        model.train()
        for i, data in enumerate(trainloader):
            images, targets = data
            images = images.to(device)
            targets = targets.to(device)           
            optimizer.zero_grad()
            outputs = model(images)
            loss_seg = loss_criteria(outputs, targets)
            f1.write( '%d %f\n' % (epoch, loss_seg.item()) )
            # loss_arr.append(loss_seg.item())
            loss_seg.backward()
            optimizer.step()
            #print("paso1")

        # -----------------------Validation------------------------------------
        # no update parameter gradients during validation
        with torch.no_grad():
            #print("paso2")
            for data_val in valloader:
                images_val, targets_val = data_val
                model.eval()
                images_val = images_val.to(device)
                targets_val = targets_val.to(device)
                #print("paso3")
                outputs_val = model(images_val)
                _, predicted = torch.max(outputs_val.data, 1)
                # ----------Compute dice-----------
                predicted_val = predicted.data.cpu().numpy()
                targets_val = targets_val.data.cpu().numpy()
                dsc = []
                
                #print("lista dsc creada")
                for i in range(1, num_classes):  # ignore Background 0
                    dsc_i,_,_,_= (dice(predicted_val, targets_val, i))
                    dsc.append(dsc_i)
                dsc_comp = dsc
                dsc = np.mean(dsc)
                
                

        #-------------------Debug-------------------------
        for param_group in optimizer.param_groups:
            print('%19.7f | %0.7f | %7d | %14.9f | %14.9f |' % (\
                    time.perf_counter(), param_group['lr'], epoch, loss_seg.item(), dsc))
            f2.write('%19.7f | %0.7f | %7d | %14.9f | %14.9f | %14.9f | %14.9f | %14.9f |\n' % (\
                    time.perf_counter(), param_group['lr'], epoch, loss_seg.item(), dsc,dsc_comp[0],dsc_comp[1],dsc_comp[2]))

        #Save checkpoint
        if (epoch % 100) == 0 or epoch == (num_epoch - 1) or (epoch % 1000) == 0:
            torch.save(model.state_dict(), './checkpoints/' + '%s_%s.pth' % (checkpoint_name, str(epoch).zfill(5)))
    
    
    f2.flush()
    f1.close()
    f3 = open("D:/EduardoCavieres/TesisMagister/Udense_RN_128/checkpoints/last_train.txt" , "w")
    f3.write("inicio_entrenamiento:%s \n" %hora)
    
    f_log = open("net_log.txt","a+") 
    f_log.write("%s \t ,train_start \n" %hora)

    hora = datetime.now()
    hora = hora.strftime(("%d-%m-%Y_%H-%M-%S"))	
    
    f3.write("fin_entrenamiento:%s \n" %hora)
    f3.close()
    f2.write( 'Fecha fin: %s \n' %hora )
    f2.close()
    
    f_log.write("%s \t ,train_end \n" %hora)
    f_log.close()



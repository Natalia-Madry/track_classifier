import logging, torch, random, time
from read_data import load_downstream_tracks, calc_avg_ut_xy_position
from load_data import TrackData
from torch_geometric.loader import DataLoader
from model_pm import TrackMessPassMod
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def format_time(full_t_seconds):
    t_hours=int(full_t_seconds//3600)
    t_minutes=int((full_t_seconds%3600)//60)
    t_seconds=full_t_seconds%60
    return f"{t_hours:02d}:{t_minutes:02d}:{t_seconds:06.3f}"

start_time=time.time()

logging.basicConfig(level=logging.INFO, format='%(message)s')
logging.info("\n--------------------------------------------------------------------------------------")
logging.info("-------------------------------- TRACK CLASSIFIER ------------------------------------\n")
logging.info("\n----------------------------- STARTED DATA LOADING ---------------------------------\n")

load_start_t=time.time()
events = load_downstream_tracks("complete_info_downstream_data.csv")
events = calc_avg_ut_xy_position(events)
random.shuffle(events)

n=len(events)
n_train=int(0.8*n) #80% for training, 20 % for testing
train_events=events[:n_train]
test_events=events[n_train:]

load_end_t=time.time()
load_time=format_time(load_end_t-load_start_t)
logging.info(f"Load stage runtime: {load_time}")
logging.info("\n------------------------------ ENDED DATA LOADING ----------------------------------\n")

logging.info("---------------------------- STARTED GRAPH BUILDING --------------------------------\n")
graph_b_start_t=time.time()
train_dataset=TrackData(train_events)
test_dataset=TrackData(test_events)
loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader=DataLoader(test_dataset, batch_size=8, shuffle=True)
logging.info(f"Loaded {len(train_dataset)} training tracks and {len(test_dataset)} test tracks.")
graph_b_end_t=time.time()
graph_b_time=format_time(graph_b_end_t-graph_b_start_t)
logging.info(f"Graph building stage runtime: {load_time}")

logging.info("----------------------------- ENDED GRAPH BUILDING ---------------------------------\n")


hit_chr=7
neurons=24

labels = torch.cat([data.y for data in train_dataset]).view(-1)

#used with BCEWLL(pos_weight)
num_pos=(labels==1).sum().item()
num_neg=(labels==0).sum().item()

#weight of classes (we have more true tracks than false ones)
pos_w=torch.tensor([num_neg/num_pos], dtype=torch.float)
# logging.info(f"Class balance: pos={num_pos}, neg={num_neg}, pos_weight={pos_weight.item():3f}")

model =TrackMessPassMod(hit_chr, neurons)
#criterion: loss function - how much wrong is model prediction
#BCE = Binary Cross Entropy (because we have binary problem (true/false))
# criterion=nn.BCEWithLogitsLoss()
criterion=nn.BCEWithLogitsLoss(pos_weight=pos_w)


#optimizer: optimizing weight of networks
optimizer=torch.optim.Adam(model.parameters(), lr=1e-3)

#training
logging.info("\n----------------------------- STARTED TRAINING ---------------------------------\n")
training_start_t=time.time()

train_losses=[]
epoch_times=[]
epochs=40
for epoch in range(epochs):
    epoch_start_t=time.time()
    model.train()
    total_loss=0.0
    for batch in loader:
        optimizer.zero_grad()
        logits=model(batch)
        labels=batch.y.view(-1).float()
        # logging.info(f"\nlogits: {logits[:].detach()}")
        # logging.info(f"labels: {labels[:]}\n")
        loss=criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()*labels.size(0)
    avg_loss=total_loss/len(train_dataset)
    train_losses.append(avg_loss)
    epoch_end_t=time.time()
    epoch_time=epoch_end_t-epoch_start_t
    epoch_times.append(epoch_time)
    logging.info(f"------------------------------------------------------------------\
                 \nEpoch {epoch:02d} | loss={avg_loss:.4f} | time={format_time(epoch_time)}")
training_end_t=time.time()
training_time=format_time(training_end_t-training_start_t)
avg_epoch_time=sum(epoch_times)/len(epoch_times)
logging.info(f"Mean epoch time: {format_time(avg_epoch_time)}")
logging.info("----------------------------- TRAINING ENDED ---------------------------------\n")
logging.info(f"Train stage runtime: {training_time}")

logging.info("----------------------------- STARTED TESTING ---------------------------------\n")
test_start_t=time.time()
model.eval()

#no grad - no training
with torch.no_grad():
    all_probs = [] #all tracks
    all_labels = []

    for batch in test_loader:
        logits = model(batch) #logit: number at the end of model (-inf, inf)
        probs = torch.sigmoid(logits) #change logit to probability (0, 1)
        labels = batch.y.view(-1)

        all_probs.append(probs)
        all_labels.append(labels)

all_probs = torch.cat(all_probs)
all_labels = torch.cat(all_labels)
test_end_t=time.time()
test_time=format_time(test_end_t-test_start_t)
logging.info("\n-------------------------------- TESTING ENDED -------------------------------------\n")
logging.info(f"Test stage runtime: {test_time}")

print("\nThresholds scan")
for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    preds = (all_probs > thr).long()
    #confusion matrix
    #tp - found true tracks (true positive)
    #tn - missed true track (true negative)
    #fp - wrong prediction to true (false positive)
    #fn - false negative
    tp=((preds== 1)&(all_labels==1)).sum().item()
    tn=((preds==0)&(all_labels==0)).sum().item()
    fp=((preds==1)&(all_labels==0)).sum().item()
    fn=((preds==0)&(all_labels==1)).sum().item()

    recall =tp/(tp+fn+1e-8)
    precision =tp/(tp+fp+1e-8)

    logging.info(
        f"thr={thr:.1f} | "
        f"TP={tp:3d} FP={fp:3d} FN={fn:3d} TN={tn:3d} | "
        f"rec={recall:.3f} prec={precision:.3f}"
    )
    #add confusion matrix

probs_downstream=all_probs[all_labels==1].numpy()
probs_ghost=all_probs[all_labels==0].numpy()

#ENG
# plt.figure()
# plt.hist(probs_downstream, bins=30, label="Downstream tracks",alpha=0.6, density=True)
# plt.hist(probs_ghost, bins=30, label="Ghosts",alpha=0.6, density=True)
# plt.xlabel("Predicted probability")
# plt.ylabel("Density")
# plt.title("Probability distribution")
# plt.legend()
# plt.show()

#train losses chart
plt.figure(figsize=(7, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o", color="teal")
plt.xlabel("Epoka", fontsize=11)
plt.ylabel("Wartość funkcji straty", fontsize=11)
plt.title("Przebieg uczenia modelu", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, alpha=0.3)
plt.show()

#PL
plt.figure(figsize=(7, 5))
plt.hist(probs_downstream, bins=30, label="Tory prawdziwe",alpha=0.7, density=True, color="teal", linewidth=1.5)
plt.hist(probs_ghost, bins=30, label="Tory fałszywe",alpha=0.5, density=True, color="crimson",edgecolor="crimson",hatch="///", linewidth=1)
plt.xlabel("Przewidywane prawdopodobieństwo", fontsize=11)
plt.ylabel("Gęstość prawdopodobieństwa", fontsize=11)
plt.title("Rozkład prawdopodobieństwa", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.show()

thr=0.7
preds = (all_probs > thr).long()
#confusion matrix for threshold
tp=((preds== 1)&(all_labels==1)).sum().item()
tn=((preds==0)&(all_labels==0)).sum().item()
fp=((preds==1)&(all_labels==0)).sum().item()
fn=((preds==0)&(all_labels==1)).sum().item()

#confusion matrix
cm=np.array([[tp, fn], [fp, tn]])
cm_norm=cm/cm.sum(axis=1, keepdims=True)

#PL
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Tory prawdziwe", "Tory fałszywe"], fontsize=10)
ax.set_yticklabels(["Tory prawdziwe", "Tory fałszywe"], fontsize=10)
ax.set_xlabel("Predykcja modelu", fontsize=11)
ax.set_ylabel("Dane rzeczywiste", fontsize=11)
ax.set_title("Znormalizowana macierz pomyłek", fontsize=12)
for i in range(2):
    for j in range(2):
        ax.text(
            j, i, f"{cm_norm[i, j]:.2f}",
            ha="center", va="center",
            color="black", fontsize=11
        )
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Udział przypadków", fontsize=10)
plt.show()

total = len(all_labels)

true_correct=((preds==1) & (all_labels==1)).sum().item()
ghost_correct= ((preds==0) & (all_labels== 0)).sum().item()

true_total=(all_labels==1).sum().item()
ghost_total=(all_labels==0).sum().item()

print(f"True tracks: {true_correct}/{true_total} correctly classified.\
      \nRatio correct: {true_correct/true_total}\n")
print(f"\nGhost tracks: {ghost_correct}/{ghost_total} correctly classified.\
      \nRatio false: {ghost_correct/ghost_total}\n")
logging.info("\n------------------------------------------------------------------------------------")
end_time=time.time()
runtime=format_time(end_time-start_time)
logging.info(f"Total program runtime: {runtime}")


import logging
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from build_graph import build_graph_from_track
from read_data import load_downstream_tracks, calc_avg_ut_xy_position

logging.basicConfig(level=logging.INFO, format='%(message)s')

class TrackData(Dataset): #TrackData - changes track to torch geometric graph
    def __init__(self, events):
        super().__init__() #Run the Dataset constructor
        self.events=events

    def len(self):
        return len(self.events)

    def get(self, id):
        track=self.events[id]
        return build_graph_from_track(track)

if __name__=="__main__":
    events=load_downstream_tracks("sample_small_new_data.csv")
    events=calc_avg_ut_xy_position(events)

    batch_size=6
    data =TrackData(events)
    data_loader=DataLoader(data, batch_size)

    logging.info("Loaded data converted to graphs: ")
    for batch in data_loader:
        logging.info(batch)






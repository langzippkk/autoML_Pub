| File     | Description         |
| ---------- | --------------------------------------------------------------- |
| embeddings.py | Obtain the embedding results of oboe output pipeline, autosklearn output pipeline, metadata of the dataset, and TPOT output pipeline separately. |
| metric_neural_network.py| Feed the constructed neural network separately with metadata_embedding, oboe_embedding, autosklearn_embedding, tpot_embedding, oboe_metadata_embedding, autosklearn_metadata_embedding, tpot_metadata_embedding and all, obtain the distance metrix, and implement argmin function. Finally, we can get the predicted pipeline from the competition with the index obtained from argmin. |
| model.h5 | Checkpoint of one specific run of the neural network. |

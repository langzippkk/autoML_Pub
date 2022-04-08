| File     | Description         |
| ---------- | --------------------------------------------------------------- |
| single_oboe.py | Obtain the output pipeline and accuracy of pipeline from Oboe. |
| autosklearn.py | Obtain the output pipeline and accuracy of pipeline from Autosklearn. |
| TPOT.py | Obtain the output pipeline and accuracy of pipeline from TPOT. |
| alphad3m.py | Obtain the output pipeline and accuracy of pipeline from alphad3m. |
| embeddings.py | Obtain the embedding results of oboe output pipeline, autosklearn output pipeline, metadata of the dataset, and TPOT output pipeline separately. |
| metric_neural_network.py| Feed the constructed neural network separately with metadata_embedding, oboe_embedding, autosklearn_embedding, tpot_embedding, oboe_metadata_embedding, autosklearn_metadata_embedding, tpot_metadata_embedding and all, obtain the distance metrix, and implement argmin function. Finally, we can get the predicted pipeline from the competition with the index obtained from argmin. |
| estimation_executionengine.py | Obtain accuracy of the predicted pipeline when implementing the pipeline on a specific dataset in execution engine. |
| Euclidean_distance_prediction.py | Obtain the predicted pipeline by implementing Euclidean distance prediction solution and locating the most similar dataset.|












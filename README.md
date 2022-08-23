RDGNN
---

## TODO:

- Improve dataloader returning samples

- **Improve code structure, make things easier to use, and more modular.**

- Improve object processing/implicit ordering. 

- Improve data collection, reduce size

- Fix currently unused self-relations

- Fix data-collection task labeling

- **Add offline testing**

- Add online planning through ros

- Find out how the one hot encoding encoding network is being trained

- Figure out why/what total_loss += self.bce_loss(outs_decoder_2_edge['pred_sigmoid'][:], x_tensor_dict_next['batch_all_obj_pair_relation'][:, :]) is doing. 

device = next(model.parameters()).device
        model.eval()

        indices = [1,2,3,4,5]

        with torch.no_grad():
            for idx in indices:
                item = self.eval_dataset[idx] # row
                inputs = {
                    'input_ids': item['input_ids'].unsqueeze(0).to(device),
                    'attention_mask': item['attention_mask'].unsqueeze(0).to(device)
                }

                outputs = model(**inputs)

                if isinstance(outputs, torch.Tensor):
                    logits_v = outputs[:, :32]
                    logits_a = outputs[:, 32:]
                else:
                    logits_v, logits_a = outputs

                pred_bin_v = torch.argmax(logits_v, dim=1).item()
                pred_bin_a = torch.argmax(logits_a, dim=1).item()

                pred_v = (pred_bin_v * 0.25) + 1.125
                pred_a = (pred_bin_a * 0.25) + 1.125

                real_bin_v = item['labels'][0].item()
                real_bin_a = item['labels'][1].item()
                real_v = (real_bin_v * 0.25) + 1.125
                real_a = (real_bin_a * 0.25) + 1.125

                text = self.tokenizer.decode(item['input_ids'] , skip_special_tokens = True)

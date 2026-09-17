"""FGSM at raw word embeddings, with padding masked and no second embedding transform."""
import torch
from .model import loss


def perturb(model, batch, cfg):
    epsilon = float(cfg['m4']['epsilon'])
    if epsilon < 0:
        raise ValueError('epsilon must be nonnegative')
    with torch.enable_grad():
        embeddings = model.encoder.get_input_embeddings()(batch['input_ids']).detach().requires_grad_(True)
        output = model(inputs_embeds=embeddings, attention_mask=batch['attention_mask'])
        value = loss(output, batch['targets'], batch['observed'], model.tasks, cfg['task_weights'])
        grad = torch.autograd.grad(value, embeddings, create_graph=False)[0]
        delta = epsilon * grad.sign() * batch['attention_mask'].unsqueeze(-1)
    return (embeddings.detach() + delta.detach()).detach(), delta.detach()


def adversarial_loss(model, batch, cfg):
    adv, _ = perturb(model, batch, cfg)
    clean = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
    perturbed = model(inputs_embeds=adv, attention_mask=batch['attention_mask'])
    return (loss(clean, batch['targets'], batch['observed'], model.tasks, cfg['task_weights'])
            + float(cfg['m4']['lambda_adv']) * loss(perturbed, batch['targets'], batch['observed'], model.tasks, cfg['task_weights']))

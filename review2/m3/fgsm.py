"""FGSM at raw word embeddings; adapted from validated/adversarial.py for token labels."""
import torch
from review2.m2.loss import token_loss


def perturb(model, batch, epsilon, weights=None):
    if epsilon < 0:
        raise ValueError('epsilon must be nonnegative')
    embeddings = model.get_input_embeddings()(batch['input_ids']).detach().requires_grad_(True)
    output = model(inputs_embeds=embeddings, attention_mask=batch['attention_mask']).logits
    loss = token_loss(output, batch['labels'], weights)
    gradient = torch.autograd.grad(loss, embeddings)[0]
    mask = (batch['attention_mask'].bool() & (batch['labels'] != -100)).unsqueeze(-1)
    delta = epsilon * gradient.sign() * mask
    return embeddings.detach() + delta.detach(), delta.detach()


def adversarial_loss(model, batch, epsilon, lambda_adv, weights=None):
    if lambda_adv < 0:
        raise ValueError('lambda_adv must be nonnegative')
    clean = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
    clean_loss = token_loss(clean, batch['labels'], weights)
    if lambda_adv == 0:
        return clean_loss
    embeddings, _ = perturb(model, batch, epsilon, weights)
    adversarial = model(inputs_embeds=embeddings, attention_mask=batch['attention_mask']).logits
    return clean_loss + lambda_adv * token_loss(adversarial, batch['labels'], weights)

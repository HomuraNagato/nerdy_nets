
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained('gpt2')

eg1 = "The Manhattan bridge Uder332f Reconstruc'ed"

eg2 = "I think it should be fixed on either UTC standard or"

eg3 = "<|endoftext|> year <|endoftext|> with the current zone <|endoftext|> Moving <|endoftext|> add a lot of <|endoftext|> to the <|endoftext|> of <|endoftext|> <|endoftext|> and have <|endoftext|> <|endoftext|> I think <|endoftext|> <|endoftext|> time made sense in the <|endoftext|> <|endoftext|> when <|endoftext|> was more <|endoftext|> and <|endoftext|> light was <|endoftext|> and often <|endoftext|> Now we have <|endoftext|> that work <|endoftext|> with simple <|endoftext|> <|endoftext|> and <|endoftext|> more <|endoftext|> to <|endoftext|> a small amount on energy for <|endoftext|> and save the <|endoftext|> cost of engineering things to work with the complex <|endoftext|> <|endoftext|> as well as saving the <|endoftext|> to <|endoftext|> <|endoftext|> has gotten much more efficient over <|endoftext|> we can <|endoftext|> out a lot more <|endoftext|> per unit of energy from a 2012 <|endoftext|> or LED than a <|endoftext|> could in <|endoftext|> or a <|endoftext|> could in <|endoftext|> <|endoftext|> a lot of room for <|endoftext|> in how we use lights as <|endoftext|> as lighting control gets more <|endoftext|> there will be a lot of <|endoftext|> from not <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> time is no <|endoftext|> worth <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|>"

generated = tokenizer.encode(eg2)
context = torch.tensor([generated])
past = None

for i in range(100):
    #print("context", context, context.shape) # context is id of word
    output, past = model(context, past=past)
    print("output being argmaxed", output.shape, "\t", output[0,:])
    token = torch.argmax(output[0, :])
    print("token taken from argmax", token)
    generated += [token.tolist()]
    context = token.unsqueeze(0)

sequence = tokenizer.decode(generated)

print(sequence)

import torch
from sklearn import manifold
import plotly.graph_objects as go


def preprocessing_for_bert(data, tokenizer_obj, max_len=300):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    @return   attention_masks_without_special_tok (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model excluding the special tokens (CLS/SEP)
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer_obj.encode_plus(
            text=sent,  # Preprocess sentence
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=max_len,  # Max length to truncate/pad
            padding="max_length",  # Pad sentence to max length
            truncation=True,  # Truncate longer seq to max_len
            return_attention_mask=True,  # Return attention mask
        )

        # Add the outputs to the lists
        input_ids.append(encoded_sent.get("input_ids"))
        attention_masks.append(encoded_sent.get("attention_mask"))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    # lets create another mask that will be useful when we want to average all word vectors later
    # we would like to average across all word vectors in a sentence, but excluding the CLS and SEP token
    # create a copy
    attention_masks_without_special_tok = attention_masks.clone().detach()

    # set the CLS token index to 0 for all sentences
    attention_masks_without_special_tok[:, 0] = 0

    # get sentence lengths and use that to set those indices to 0 for each length
    # essentially, the last index for each sentence, which is the SEP token
    sent_len = attention_masks_without_special_tok.sum(1).tolist()

    # column indices to set to zero
    col_idx = torch.LongTensor(sent_len)
    # row indices for all rows
    row_idx = torch.arange(attention_masks.size(0)).long()

    # set the SEP indices for each sentence token to zero
    attention_masks_without_special_tok[row_idx, col_idx] = 0

    return input_ids, attention_masks, attention_masks_without_special_tok


def get_preds(sentences, tokenizer_obj, model_obj):
    """
    Quick function to extract hidden states and masks from the sentences and model passed
    """
    # Run the sentences through tokenizer
    input_ids, att_msks, attention_masks_wo_special_tok = preprocessing_for_bert(
        sentences, tokenizer_obj
    )
    # Run the sentences through the model
    outputs = model_obj(
        input_ids,
        att_msks,
    )

    # Lengths of each sentence
    sent_lens = att_msks.sum(1).tolist()

    # calculate unique vocab
    # #get the tokenized version of each sentence (text form, to label things in the plot)
    tokenized_sents = [tokenizer_obj.convert_ids_to_tokens(i) for i in input_ids]
    return {
        "hidden_states": outputs.hidden_states,
        "pooled_output": outputs.pooler_output,
        "attention_masks": att_msks,
        "attention_masks_without_special_tok": attention_masks_wo_special_tok,
        "tokenized_sents": tokenized_sents,
        "sentences": sentences,
        "sent_lengths": sent_lens,
    }


def plt_dists(
    dists,
    sentences_and_labels,
    dims=2,
    title="",
    xrange=[-0.5, 0.5],
    yrange=[-0.5, 0.5],
    zrange=[-0.5, 0.5],
):
    """
    Plot distances using MDS in 2D/3D
    dists: precomputed distance matrix
    sentences_and_labels: tuples of sentence and label_ids
    dims: 2/3 for 2 or 3 dimensional plot, defaults to 2 for any other value passed
    words_of_interest: list of words to highlight with a different color
    title: title for the plot
    """
    # get the sentence text and labels to pass to the plot
    sents, color = zip(*sentences_and_labels)

    # https://community.plotly.com/t/plotly-colours-list/11730/6
    colorscale = [
        [0, "deeppink"],
        [1, "yellow"],
    ]  # , [2, 'greens'], [3, 'reds'], [4, 'blues']]

    # dists is precomputed using cosine similarity/other other metric and passed
    # calculate MDS with number of dims passed
    mds = manifold.MDS(
        n_components=dims, dissimilarity="precomputed", random_state=60, max_iter=90000
    )
    results = mds.fit(dists)

    # get coodinates for each point
    coords = results.embedding_

    # plot 3d/2d
    if dims == 3:
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    z=coords[:, 2],
                    mode="markers+text",
                    textposition="top center",
                    text=sents,
                    marker=dict(
                        size=12, color=color, colorscale=colorscale, opacity=0.8
                    ),
                )
            ]
        )
    else:
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    text=sents,
                    textposition="top center",
                    mode="markers+text",
                    marker=dict(
                        size=12, color=color, colorscale=colorscale, opacity=0.8
                    ),
                )
            ]
        )

    fig.update_layout(template="plotly_dark")
    if title != "":
        fig.update_layout(title_text=title)
        fig.update_layout(
            titlefont=dict(
                family="Courier New, monospace", size=14, color="cornflowerblue"
            )
        )

    # update the axes ranges
    fig.update_layout(yaxis=dict(range=yrange))
    fig.update_layout(xaxis=dict(range=xrange))
    fig.update_traces(textfont_size=10)

    # TO DO: fix this. I could not get this to work. somehow the library does not like the zaxis.
    # if dims==3:
    # fig.update_layout(zaxis=dict(range=zrange))
    fig.show()


def get_word_vectors(
    hidden_layers_form_arch, token_index=None, mode="average", top_n_layers=4
):
    """
    retrieve vectors for all tokens from the top n layers and return a concatenated, averaged or summed vector
    hidden_layers_form_arch: tuple returned by the transformer library
    token_index: None/Index:
      If None: Returns all the tokens
      If Index: Returns vectors for that index in each sentence

    mode=
          'average' : avg last n layers
          'concat': concatenate last n layers
          'sum' : sum last n layers
          'last': return embeddings only from last layer
          'second_last': return embeddings only from second last layer

    top_n_layers: number of top layers to concatenate/ average / sum
    """

    vecs = None
    if mode == "concat":
        vecs = torch.cat(hidden_layers_form_arch[-top_n_layers:], dim=2)

    if mode == "average":
        vecs = torch.stack(hidden_layers_form_arch[-top_n_layers:]).mean(0)

    if mode == "sum":
        vecs = torch.stack(hidden_layers_form_arch[-top_n_layers:]).sum(0)

    if mode == "last":
        vecs = hidden_layers_form_arch[-1:][0]

    if mode == "second_last":
        vecs = hidden_layers_form_arch[-2:-1][0]

    if vecs is not None and token_index:
        # if a token index is passed, return values for a particular index in the sequence instead of vectors for all
        return vecs.permute(1, 0, 2)[token_index]
    return vecs


def get_sent_vectors(input_states, att_mask):
    """
    get a sentence vector by averaging over all word vectors -> this could come from any layers or averaged themselves (see get_all_token_vectors function)
    input_states: [batch_size x seq_len x vector_dims] -> e.g. results from  hidden stats from a particular layer
    att_mask: attention mask passed should have already maseked the special tokens too i.e. CLS/SEP/<s>/special tokens masked out with 0 -> [batch_size x max_seq_length]
    ref: https://stackoverflow.com/questions/61956893/how-to-mask-a-3d-tensor-with-2d-mask-and-keep-the-dimensions-of-original-vector
    """

    # print(input_states.shape) #-> [batch_size x seq_len x vector_dim]

    # Let's get sentence lengths for each sentence
    sent_lengths = att_mask.sum(
        1
    )  # att_mask has a 1 against each valid token and 0 otherwise

    # create a new 3rd dim and broadcast the attention mask across it -> this will allow us to use this mask with the 3d tensor input_hidden_states
    att_mask_ = att_mask.unsqueeze(-1).expand(input_states.size())

    # use mask to 0 out all the values against special tokens like CLS, SEP , <s> using mask
    masked_states = input_states * att_mask_

    # calculate average
    sums = masked_states.sum(1)
    avg = sums / sent_lengths[:, None]
    return avg


def get_sentence_vectors(
    model_output,
    sentences,
    wrd_vec_mode="concat",
    wrd_vec_top_n_layers=4,
    viz_dims=2,
    sentence_emb_mode="average_word_vectors",
    title_prefix=None,
    plt_xrange=[-0.05, 0.05],
    plt_yrange=[-0.05, 0.05],
    plt_zrange=[-0.05, 0.05],
):
    """
    Get vectors for all sentences and visualize them based on cosine distance between them

    model_output: model results extracted as a dictionary from get_preds function
    sentences_and_labels: tuple of sentence and labels_ids
    att_msk: attention mask that also marks the special tokens (CLS/SEP etc.) as 0
    mode=
          'average' : avg last n layers
          'concat': concatenate last n layers
          'sum' : sum last n layers
          'last': return embeddings only from last layer
          'second_last': return embeddings only from second last layer
    viz_dims:2/3 for 2D/3D plot
    title_prefix: String to add before the descriptive title. Can be used to add model name etc.
    """
    title_wrd_emv = "{} across {} layers".format(wrd_vec_mode, wrd_vec_top_n_layers)

    # get word vectors for all words in the sentence
    if sentence_emb_mode == "average_word_vectors":
        title_sent_emb = (
            "average(word vectors in the sentence); Sentence Distance: Cosine"
        )
        word_vecs_across_sent = get_word_vectors(
            model_output["hidden_states"],
            mode=wrd_vec_mode,
            token_index=None,
            top_n_layers=wrd_vec_top_n_layers,
        )  # returns [batch_size x seq_len x vector_dim]
        sent_vecs = get_sent_vectors(
            word_vecs_across_sent, model_output["attention_masks_without_special_tok"]
        )
    else:
        title_sent_emb = "First tok (CLS) vector; Sentence Distance: Cosine"
        # Get the pooled results from the first token (e.g. CLS token in case of BERT)

        # Note from https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        # This results is usually not a good summary of the semantic content of the
        # input, you’re often better with averaging or
        # pooling the sequence of hidden-states for the whole input sequence.
        sent_vecs = model_output["pooled_output"]  # vector

    if title_prefix:
        final_title = "{} Word Vec: {}; Sentence Vector: {}".format(
            title_prefix, title_wrd_emv, title_sent_emb
        )
    else:
        final_title = "Word Vec: {}; Sentence Vector: {}".format(
            title_wrd_emv, title_sent_emb
        )
    mat = sent_vecs.detach().numpy()
    return mat

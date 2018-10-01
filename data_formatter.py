import pickle
def format_gts(vid_to_caption_map, id_to_word, mode):
    temp = []
    for video_id in vid_to_caption_map.keys():
        for caption in vid_to_caption_map[video_id]:
            processed_caption = " ".join([id_to_word[id] for id in caption[1:]])
            temp.append({'image_id' : video_id, 'caption' : processed_caption})
    print(len(temp))
    gts = {}
    gts["annotations"] = temp
    with open('./meta_data/cider_jsons/gts_{}.pkl'.format(mode), "wb") as file:
        pickle.dump(gts, file)


if __name__ == '__main__':
    modes = ['train', 'test', 'valid']

    vocab_path = "./meta_data/vocab.pkl"
    vocab = pickle.load(open(vocab_path,'rb'))
    id_to_word = {v: k for k, v in vocab.items()}

    for mode in modes:
        with open('./meta_data/{}_vid_to_caption.pkl'.format(mode), "rb") as file:
            vc_map = pickle.load(file)
            format_gts(vc_map, id_to_word, mode)

from transformers import pipeline, BartForConditionalGeneration, BartTokenizer


# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
# summarizer.save_pretrained("/home/gautam/Documents/wspace/video_Search/models1")

summarizer = BartForConditionalGeneration.from_pretrained("/home/gautam/Documents/wspace/video_Search/models1")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

summarizer_pipeline = pipeline("summarization", model=summarizer, tokenizer=tokenizer)

article = """
Before 2015, YouTube employed a matric factorization approach in order to train its model with only users’ watch history. This approach is redefined in the current system to accept generalized inputs, i.e., continuous and categorical features. The viewer’s demographics such as gender, age, logged-in state are a few examples of continuous features. While embedded search and watch histories are considered to be categorical features. These inputs are passed on to four densely connected ReLU layers. During training, the inputs are then forwarded to a softmax layer that classifies the videos into different categories.

Apart from the regular user-related features, the system also integrates several other aspects into features and training sets based on certain cases."""
max_len = 100
min_len = 50

summ = summarizer_pipeline(article, max_length=max_len, min_length=min_len, do_sample=False)

print(summ[0]['summary_text'])





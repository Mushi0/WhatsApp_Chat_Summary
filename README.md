# WhatsApp_Chat_Summery
For Parsing WhatsApp chat history

inspired by: https://foolishfox.cn/posts/202402-WeChatMsgAnalysis.html


## How to use

### setup:
```
poetry env use 3.10.7
poetry install
```

### run the script:
```
python main.py <chat_log_downloaded_from_WhatsApp> <path_to_save_result> <name_1_on_WhatsApp> <name_2_on_WhatsApp> <name_1_in_result> <name_2_in_result> <year>
```

example:
```
python main.py chat.txt image_2023/ "John Smith" "Alice Smith" John Alice 
```
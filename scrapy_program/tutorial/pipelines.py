# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html


#class TutorialPipeline(object):
#    def process_item(self, item, spider):
#        return item

from w3lib.html import remove_tags
import re
import string

def clense(text, space_replacer = ' ', to_lower = True, remove_punc = True):
    # remove HTML comments first as suggested in https://stackoverflow.com/questions/28208186/how-to-remove-html-comments-using-regex-in-python
    text = re.sub("(<!--.*?-->)", "", text, flags=re.DOTALL)
    text = remove_tags(text)
    text = re.sub(r'[^\x00-\x7F]+',' ', text)   #remove non-ascii characters
    text = text.replace("&amp;", "and")
    text = text.replace("&", "and")
    text.strip()
    text.rstrip()
    text = text.replace("\r\n", "")
    if to_lower:
        text = text.lower()

    if remove_punc:
        # from https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python
        text = re.sub(r'[^\w\s]', '', text)   #remove punctuation marks and non-word
        text = text.replace(",", "")

    text = re.sub(' +', space_replacer, text)
    text = text.replace("\"", "")
    #if  all(ord(char) < 128 for char in text) == False:
    #    text = ''
    ''.join(i for i in text if ord(i)<128)
    return text

class BHRRCPipeline(object):

    def process_item(self, item, spider):
        i = 0
        new_content = ''
        new_summary = ''
        for content in item['content']:
            content = clense(content,to_lower=False,remove_punc=False)
            if i==0:
                new_summary = content.strip()
            if i>0:
                new_content = new_content + " " + content
            i = i+1
        item['content'] = new_content.strip()
        item['summary'] = new_summary.strip()

        new_title = ''
        for title in item['title']:
            new_title = new_title + clense(title,to_lower=False,remove_punc=False)
        item['title'] = new_title.strip()

        new_author = ''
        for author in item['author']:
            new_author = new_author + clense(author,to_lower=False,remove_punc=False)

        new_author = new_author[8:]
        new_author = new_author[:-17]
        item['author'] = new_author.strip()

        return item

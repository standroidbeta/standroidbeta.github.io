---
layout: post
title: Biking Down the Econmic Growth Trail with Data!
subtitle: How Greenville, South Carolina's Swamp Rabbit Trail Might Be Stimulating Economic Growth
bigimg: /img/Greenville_SC_Downtown_Riverwalk_Panoramic_View.jpg
tags: [Greenville South Carolina, Economics, Swamp Rabbit Trail]
---

## Purpose
This is my first data science project where I have compiled data Greenville, SC’s Swamp Rabbit Trail Survey data and Industry Growth by Employees in the workforce to evaluate any potential for correlation of having a bike/walk trail can stimulate economic growth for a Metro area.

## A little background
Greenville South Carolina is located in near the upper northwest of the state. It is the largest county in the state and is growing both in population and economically. Greenville’s growth and influence has been a stimulating force to its surrounding city and county neighbors as well.

![SRT Map](/img/SRT-Map.png)

One of factors in Greenville’s growth is the the Swamp Rabbit Trail. The Swamp Rabbit Trail is a 22 mile multi-use (walking & bicycling) connected greenway network that traverses along the Reedy River. It is an old railroad corridor that connects Greenville city parks to Travelers Rest.

The trail opened up in 2009 and is one of the most popular recreational attractions in the Greenville Metro area. It has been so inspirational that the surrounding counties of Pickens, Anderson, and Spartanburg have also implemented  either bike path or trails. Both the Doodle Trail in Pickens County  connecting the cities Easley to Pickens and the Swamp Rabbit Trail were inspired by the Rails to Trails initiative where communities are repurposing no-longer used train trail routes and converting them to bike/walk trails. Spartanburg has established a in-town bike path system called Bike Town that has a bicycle sharing program call B-Share.

## Here are some data points

Greenville’s Swamp Rabbit Trail had a survey conducted that ran from 2011 to 2014 to analyze the community Impact. Here is an infographic from that study:

<p><img src='/img/Year 3 Impact Study Key Facts.jpg' alt='Impact Study' title='Impact Study' /></p>

Using data from the Greenvile Rec's Google Drive I created a this Dataframe:

```import matplotlib.pyplot as plt
import pandas as pd

traildata = pd.read_csv('https://raw.githubusercontent.com/standroidbeta/DS-Unit-1-Sprint-5-Data-Storytelling-Blog-Post/master/DS-Unit-1-Sprint-5-Data-Storytelling-Data/Swamp-Rabbit%20Economic-Impact-Results.csv')
traildata

traildata = traildata.drop(columns=['Users Outside of Greenville County',	'Percentage Outside of Greenville County',	
                                    'Users outside 50 mile radius of 29617',	'Percentage of Out of County Users that are outside a 50 mile Radius', 
                                    'Percentage of Total Users that are outside a 50 mile radius'])
traildata```

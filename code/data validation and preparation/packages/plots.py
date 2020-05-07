import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3
import matplotlib.pyplot as plt4
import datetime
import matplotlib.dates as mdates


def plot_mean_of_classes(profile):
    # plot loss function and log to Azure ML
    plt1.clf()
    x = [
     datetime.datetime.strptime(d, "%Y-%m-%d").date() for d in profile['date']
        ]
    y = profile['mean of classes']
    plt1.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt1.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt1.plot(x, y)
    plt1.gcf().autofmt_xdate()    
    plt1.xlabel("Date")
    plt1.ylabel("Mean of classes")
    plt1.title('Mean of classes over time')
    return plt1


def plot_std_of_classes(profile):
    plt2.clf()
    # plot loss function and log to Azure ML
    x = [
     datetime.datetime.strptime(d, "%Y-%m-%d").date() for d in profile['date']
        ]
    y = profile['standard deviation of classes']
    plt2.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt2.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt2.plot(x, y)
    plt2.gcf().autofmt_xdate()
    plt2.xlabel("Date")
    plt2.ylabel("Mean of classes")
    return plt2


def plot_average_word_length(profile):
    plt3.clf()
    # plot loss function and log to Azure ML
    x = [
     datetime.datetime.strptime(d, "%Y-%m-%d").date() for d in profile['date']
        ]
    y = profile['average word length']
    plt3.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt3.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt3.plot(x, y)
    plt3.gcf().autofmt_xdate()
    plt3.xlabel("Date")
    plt3.ylabel("average word length")
    return plt3


def plot_average_number_of_stopwords(profile):
    plt4.clf()
    # plot loss function and log to Azure ML
    x = [
     datetime.datetime.strptime(d, "%Y-%m-%d").date() for d in profile['date']
        ]
    y = profile['average number of stopwords']
    plt4.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt4.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt4.plot(x, y)
    plt4.gcf().autofmt_xdate()
    plt4.xlabel("Date")
    plt4.ylabel("average number of stopwords")
    return plt4

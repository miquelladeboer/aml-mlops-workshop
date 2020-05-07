import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
import datetime
import matplotlib.dates as mdates


def plot_mean_of_classes(profile):
    # plot loss function and log to Azure ML
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

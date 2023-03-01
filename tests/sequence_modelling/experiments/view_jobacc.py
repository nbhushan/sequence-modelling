# -*- coding: utf-8 -*-
"""
Created on Thu Sep 05 14:27:19 2013

@author: nbhushan
"""

from datetime import datetime,timedelta
import csv, logging, numpy
import pdb
logger = logging.getLogger()

def tstamp(t,refdate=datetime(2013,0o7,11,0,0,0),daysecs=24*3600):
  t = t-refdate  
  return t.days*daysecs+t.seconds

def data():   
    logger.info('Loading pwr data')
    with open('dauphine_1107to1108_powerdata.csv') as u:
        ur = csv.reader(u)
        hdr = next(ur)
        power = [float(x[0]) for x in ur]
        base= datetime(2013,0o7,11,00,00,00)
        dateList = [ base + timedelta(seconds=x) for x in range(0,len(power)) ]
        fdatelist = [tstamp(x) for x in dateList]
        Lpwr = tuple((fdatelist,power))

    logger.info('Loading job data')
    with open('dauphine_1107to1108_jobbacc.csv') as u:
        ur = csv.reader(u)
        hdr = next(ur)
        Ljob = tuple((tstamp(datetime.strptime(x[0],'%Y-%m-%d %H:%M:%S')), tstamp(datetime.strptime(x[1],'%Y-%m-%d %H:%M:%S')), int(x[2])) for x in ur)  
    return Lpwr, Ljob

def display(data,ax,mibshift=0,jobshift=0):
#pdb.set_trace() 
    def dconv(x,d0=-tstamp(datetime(1,1,1)),daysecs=24*3600.):
        return (d0+x)/daysecs
    Lpwr, Ljob = data
    logger.info('Formatting data')
    apwr = numpy.array(Lpwr)
#pdb.set_trace()
    ajob = numpy.array([(x[0], x[1], x[2]) for x in Ljob])
    logger.info('Plotting data')
    ax.plot_date(dconv(apwr[0]),apwr[1],'b')
    ax.bar(dconv(ajob[:,0]-jobshift),len(ajob)*[600],dconv(ajob[:,1]-ajob[:,0],0),0,color='c')
    for label, x in zip(ajob[:,2], dconv(ajob[:,0]-jobshift)):
        ax.annotate(
            label, 
            xy = (x, 0), xytext = (-20, 20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'r', alpha = 0.4),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))     
    for l in ax.get_xticklabels():
        l.update(dict(rotation=40,horizontalalignment='right'))    


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    from matplotlib.pyplot import figure,show
    f = figure()
    display(data(),f.add_subplot(111))
    f.tight_layout()
    print('Close the plot window to end the program.')
    show()

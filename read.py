import math
f=open('affinity_data.txt', 'r')

arr=f.readlines()

for name in arr:
    #if len(name)==8:
    #    print name
    if  len(name)>8:
         #sytem("cp -r ../"+name[0:4])
        value=name.split()
        #print (value[0]) 
        value_n=float(value[1])*(10**(-6))
        affinity=-math.log10(float(value_n))
        affinity_n=round(affinity,2)
        print (value[0]+"   "+str(affinity_n)+'\n')
        fw=open(value[0]+'.dat', 'w')
        fw.write(str(affinity_n))

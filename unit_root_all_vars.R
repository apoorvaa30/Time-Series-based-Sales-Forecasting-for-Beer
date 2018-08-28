i = 91
df = vals[,i]
gdpia<-ts(df)

# 8 25 61 69 70 - same type
# 9 10 11 35 36 - error
# 43 48 49 58 59 76 79 - ambiguous
print("##############################################################") 
print(colnames(Final_data[i]))
print("For all")
adf.test(gdpia)$p.value #null unit root
pp.test(gdpia)$p.value #null unit root
kpss.test(gdpia)$p.value #null stationary


print("For quarter")
r1<-gdpia[seq(1,168,4)]
r2<-gdpia[seq(2,168,4)]
r3<-gdpia[seq(3,168,4)]
r4<-gdpia[seq(4,168,4)]

pp.test(r1)$p.value
pp.test(r2)$p.value
pp.test(r3)$p.value
pp.test(r4)$p.value


print("For monthly")
q1<-gdpia[seq(1,168,12)]
q2<-gdpia[seq(2,168,12)]
q3<-gdpia[seq(3,168,12)]
q4<-gdpia[seq(4,168,12)]
q5<-gdpia[seq(5,168,12)]
q6<-gdpia[seq(6,168,12)]
q7<-gdpia[seq(7,168,12)]
q8<-gdpia[seq(8,168,12)]
q9<-gdpia[seq(9,168,12)]
q10<-gdpia[seq(10,168,12)]
q11<-gdpia[seq(11,168,12)]
q12<-gdpia[seq(12,168,12)]

pp.test(q1)$p.value
pp.test(q2)$p.value
pp.test(q3)$p.value
pp.test(q4)$p.value #
pp.test(q5)$p.value
pp.test(q6)$p.value
pp.test(q7)$p.value
pp.test(q8)$p.value
pp.test(q9)$p.value #
pp.test(q10)$p.value
pp.test(q11)$p.value
pp.test(q12)$p.value
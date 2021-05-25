class Bow():
    def __init__(self, x_train, y_train):
        self.bow = {}
        for index,row in x_train.iterrows():
            if row['probe'] not in self.bow:
                self.bow[row['probe']] = [set(),set(),set(),set()]
            for w in row['line']:
                if y_train.loc[index][0].isnumeric():
                    self.bow[row['probe']][(int)(y_train.loc[index][0])-1].add(w)

    def get_score(self, x_test, y_test):
        correct, wrong, noidea =0,0,0
        for index,row in x_test.iterrows():
            score_line =[0,0,0,0]
            if row['probe'] in self.bow:
                for w in row['line']:
                    for i in range(4):
                        if w in self.bow[row['probe']][i]:
                            score_line[i]+=1                
            if sum(score_line)==0 or score_line.count(max(score_line))>1:
                noidea +=1
            elif (str)(score_line.index(max(score_line))+1) == y_test.loc[index][0]:
                correct+=1
            else:
                wrong +=1
        return correct/len(x_test), wrong/len(x_test), noidea/len(x_test)
    
    def predict(self, x_test):
        answer = ['0']*len(x_test)
        j=0
        for index,row in x_test.iterrows():
            score_line =[0,0,0,0]
            if row['probe'] in self.bow:
                for w in row['line']:
                    for i in range(4):
                        if w in self.bow[row['probe']][i]:
                            score_line[i]+=1                
            if sum(score_line)!=0 and score_line.count(max(score_line))<2:
                answer[j]=(str)(score_line.index(max(score_line))+1)
            j+=1
                

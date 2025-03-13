function input=see_input(input)
        input(1:2)=[];% remove the health state and t, then reshape input as the 10*6
        input=reshape(input,[6,10]);    
        input=input';   
end
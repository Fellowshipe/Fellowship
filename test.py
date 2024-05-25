fraud_check = 'Y'
found_fraud_check = 'N'

is_fraud = fraud_check or found_fraud_check
        
print(is_fraud)
# 사기 체크
if fraud_check is not None and found_fraud_check is not None:
    if found_fraud_check == 'Y':
        is_fraud = 'Y'

print(is_fraud)
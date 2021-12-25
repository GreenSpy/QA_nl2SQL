import torch
from sqlnet.utils import *
from sqlnet.model.sqlbert import SQLBert, BertTokenizer
import argparse
import records
import os.path as osp

agg_dict = {0:"", 1:"AVG", 2:"MAX", 3:"MIN", 4:"COUNT", 5:"SUM"}
cond_op_dict = {0:">", 1:"<", 2:"==", 3:"!="}
rela_dict = {0:'', 1:' AND ', 2:' OR '}

class MYDBEngine:
    def __init__(self, fdb):
        self.db = records.Database('sqlite:///{}'.format(fdb))
        self.conn = self.db.get_connection()

    def execute(self, table_id, select_index, aggregation_index, conditions, condition_relation):
        """
        table_id: id of the queried table.
        select_index: list of selected column index, like [0,1,2]
        aggregation_index: list of aggregation function corresponding to selected column, like [0,0,0], length is equal to select_index
        conditions: [[condition column, condition operator, condition value], ...]
        condition_relation: 0 or 1 or 2
        """
        table_id = 'Table_{}'.format(table_id)

        # 条件数>1 而 条件关系为''
        if condition_relation == 0 and len(conditions) > 1:
            return 'Error1'
        # 选择列或条件列为0
        if len(select_index) == 0 or len(conditions) == 0 or len(aggregation_index) == 0:
            return 'Error2'

        condition_relation = rela_dict[condition_relation]

        select_part = ""
        for sel, agg in zip(select_index, aggregation_index):
            select_str = 'col_{}'.format(sel+1)
            agg_str = agg_dict[agg]
            if agg:
                select_part += '{}({}),'.format(agg_str, select_str)
            else:
                select_part += '({}),'.format(select_str)
        select_part = select_part[:-1]

        where_part = []
        for col_index, op, val in conditions:
            where_part.append('col_{} {} "{}"'.format(col_index+1, cond_op_dict[op], val))
        where_part = 'WHERE ' + condition_relation.join(where_part)

        query = 'SELECT {} FROM {} {}'.format(select_part, table_id, where_part)
        try:
            out = self.conn.query(query).as_dict()
        except:
            return 'Error3'

        result_set = [tuple(sorted(i.values(), key=lambda x:str(x))) for i in out]
        return result_set, query

if __name__ == '__main__':
    
	parser = argparse.ArgumentParser()

	parser.add_argument('--q', type=str) # 问题输入

	args = parser.parse_args()

	question = args.q

	batch_size = 1

	data_dir = '../data/'
	bert_model_dir = '../model/chinese-bert_chinese_wwm_pytorch/'
	restore_model_path = '../model/2500'

	result_path = '../data/result.json'
 
	question = {'table_id':'all','question':question}
	print(question)

	with open('../data/final_test.json','w') as f:
		json.dump(question,f)
 
	test_sql_path = osp.join(data_dir, 'final_test.json')
	test_table_path = osp.join(data_dir, 'final_test.tables.json')
	test_sql, test_table = load_data(test_sql_path, test_table_path)

	tokenizer = BertTokenizer.from_pretrained(bert_model_dir, do_lower_case=True)
	model = SQLBert.from_pretrained(bert_model_dir)
	model.load_state_dict(torch.load(restore_model_path, map_location='cpu'))

	predict_test(model, batch_size, test_sql, test_table, result_path, tokenizer=tokenizer)
 
	with open('../data/result.json', 'r',encoding="utf-8") as f:
		res_output = json.load(f)
		print(res_output) # sql子集输出
 
	table_id = 'all'
	select_index = res_output['sel']
	aggregation_index = res_output['agg']
	conditions = res_output['conds']
	condition_relation = res_output['cond_conn_op']
 
	engine = MYDBEngine('../data/final_test.db')
	final_res , final_sql = engine.execute(table_id, select_index, aggregation_index, conditions, condition_relation)
	

	final_sql = final_sql.replace('col_10','Index')
	final_sql = final_sql.replace('col_1','Name')
	final_sql = final_sql.replace('col_2','Sex')
	final_sql = final_sql.replace('col_3','Age')
	final_sql = final_sql.replace('col_4','Year')
	final_sql = final_sql.replace('col_5','City')
	final_sql = final_sql.replace('col_6','Sport')
	final_sql = final_sql.replace('col_7','Event')
	final_sql = final_sql.replace('col_8','Medal')
	final_sql = final_sql.replace('col_9','Region')
 
	print(final_sql) # 最终sql输出
	print(final_res[0][0]) # 最终结果输出